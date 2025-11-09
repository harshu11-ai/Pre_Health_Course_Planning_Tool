
from __future__ import annotations

from typing import List, Dict, Optional, Set, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pre‑Health D‑Plan Backend (MVP, Clean)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False to avoid preflight issues
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Utilities: Term ordering and helpers
# -----------------------------------------------------------------------------
TERM_ORDER = {"F": 0, "W": 1, "S": 2, "X": 3}  # Fall, Winter, Spring, Summer (X)

def parse_term(term: str) -> Tuple[int, str]:
    """Return (yy:int, season:str). Accepts e.g. '24F'."""
    if len(term) != 3 or term[-1] not in TERM_ORDER:
        raise ValueError(f"Invalid term code: {term}. Expected like '24F'.")
    yy = int(term[:2])
    season = term[-1]
    return yy, season

def term_sequence(start: str, count: int) -> List[str]:
    """Generate a sequence of `count` consecutive Dartmouth terms starting at `start`."""
    yy, s = parse_term(start)
    order_to_season = {v: k for k, v in TERM_ORDER.items()}
    seq = []
    cur_order = TERM_ORDER[s]
    cur_academic_year = yy  # Academic year starts with Fall term
    
    for _ in range(count):
        season = order_to_season[cur_order]
        
        # For W, S, X terms, use the next calendar year
        if season in ["W", "S", "X"]:
            display_year = cur_academic_year + 1
        else:  # Fall term
            display_year = cur_academic_year
            
        seq.append(f"{display_year:02d}{season}")
        
        # Move to next term
        cur_order = (cur_order + 1) % 4
        if cur_order == 0:  # Wrapped to Fall, increment academic year
            cur_academic_year += 1
            
    return seq

# -----------------------------------------------------------------------------
# Data Models (Pydantic)
# -----------------------------------------------------------------------------
class Course(BaseModel):
    id: str
    title: str
    dept: str
    difficulty: int = Field(ge=1, le=5)
    tags: List[str] = []
    typical_terms: List[str] = Field(default_factory=lambda: ["F", "W", "S"])  # seasons only
    notes: Optional[str] = None

class Prereq(BaseModel):
    course_id: str
    requires: List[str] = []        # must complete ALL in this list (AND logic)
    coreq: List[str] = []           # can be taken in same term (unused in MVP)
    min_grade: Optional[str] = None # not enforced in MVP

class Requirement(BaseModel):
    id: str
    description: Optional[str] = None
    options: List[List[str]] = Field(default_factory=list)  # unused in MVP

class Rule(BaseModel):
    id: str
    type: str  # 'mutual_exclusion' | 'difficulty_cap'
    severity: str = "warn"  # 'error' | 'warn'
    message: str
    course_pair: Optional[Tuple[str, str]] = None  # for mutual_exclusion
    difficulty_cap: Optional[int] = None           # per-term cap (unused; see request.difficulty_cap)

class StudentProfile(BaseModel):
    id: str
    entry_term: str  # e.g., '24F'
    graduation_terms: int = 16  # default number of terms to plan (4 years)
    max_courses_per_term: int = 2
    off_terms: List[str] = []   # absolute term codes like '25W'
    abroad_terms: List[str] = []
    incoming_credits: Dict[str, str] = {}
    courses_with_credit: List[str] = []
    eligible_courses: List[str] = []
    preferences: Dict[str, str] = {}

class TermPlan(BaseModel):
    label: str  # '24F'
    type: str = "on"  # 'on' | 'off' | 'abroad'
    courses: List[str] = []
    difficulty_sum: int = 0
    flags: List[str] = []

class Plan(BaseModel):
    student_id: str
    terms: List[TermPlan]
    score: float
    warnings: List[str] = []
    errors: List[str] = []
    explanations: List[str] = []
    optional_courses: List[str] = []

class GeneratePlanRequest(BaseModel):
    student: StudentProfile
    required_courses: List[str]
    optional_courses: List[str] = []
    prereqs: List[Prereq] = []
    rules: List[Rule] = []
    difficulty_cap: int = 7

class ValidatePlanRequest(BaseModel):
    plan: Plan
    prereqs: List[Prereq] = []
    rules: List[Rule] = []
    difficulty_cap: int = 7

# -----------------------------------------------------------------------------
# In-memory stores (MVP)
# -----------------------------------------------------------------------------
COURSES: Dict[str, Course] = {}
PREREQS: Dict[str, Prereq] = {}
RULES: Dict[str, Rule] = {}
PROFILES: Dict[str, StudentProfile] = {}

# -----------------------------------------------------------------------------
# Rule & Constraint Helpers
# -----------------------------------------------------------------------------
def can_take_in_term(course: Course, term_label: str) -> bool:
    """Check if a course is typically offered in the given term's season."""
    season = term_label[-1]
    return season in course.typical_terms or len(course.typical_terms) == 0

def is_term_allowed_for_classes(term_label: str, entry_term: str) -> bool:
    """Allow all non-summer terms. For summers, only allow Sophomore Summer.
    Sophomore Summer is entry_yy + 2 (e.g., 24F entry → 26X).
    """
    yy, season = parse_term(term_label)
    entry_yy, _ = parse_term(entry_term)
    if season != "X":
        return True
    sophomore_summer_yy = entry_yy + 2
    return yy == sophomore_summer_yy

def topo_sort_courses(targets: List[str], prereqs: Dict[str, Prereq]) -> List[str]:
    """Topologically sort required courses by hard prerequisites (AND logic)."""
    req_set = set(targets)
    graph: Dict[str, Set[str]] = {c: set() for c in req_set}
    indeg: Dict[str, int] = {c: 0 for c in req_set}
    for c in req_set:
        pr = prereqs.get(c)
        if not pr:
            continue
        for r in pr.requires:
            if r in req_set:
                graph[r].add(c)
                indeg[c] += 1
    queue = [c for c, d in indeg.items() if d == 0]
    order: List[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in graph.get(node, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(req_set):
        remaining = [c for c in req_set if c not in order]
        order.extend(remaining)
    return order

def _filter_and_expand_required(required: List[str], prereqs: Dict[str, Prereq], student: StudentProfile) -> List[str]:
    """Apply credits and eligibility rules:
    - Drop any course already credited in student.courses_with_credit
    - If eligible for CHEM 11, allow it to replace CHEM 05+06 (if present).
    """
    credited = set(student.courses_with_credit or [])
    out: List[str] = []
    for c in required:
        if c not in credited:
            out.append(c)
    
    # If student is eligible for CHEM 11, handle substitution logic
    if "CHEM 11" in (student.eligible_courses or []):
        # If CHEM 05 or CHEM 06 are present, replace them with CHEM 11
        if "CHEM 05" in out or "CHEM 06" in out:
            out = [c for c in out if c not in {"CHEM 05", "CHEM 06"}]
            if "CHEM 11" not in out:
                out.append("CHEM 11")
        # If CHEM 11 is already selected and student is eligible, keep it
        elif "CHEM 11" in out:
            # CHEM 11 is already in the list and student is eligible, so keep it
            pass
    
    return out

def violates_mutual_exclusion(term_courses: List[str], rules: List[Rule]) -> Optional[str]:
    pairs = [r.course_pair for r in rules if r.type == "mutual_exclusion" and r.course_pair]
    s = set(term_courses)
    for a, b in pairs:
        if a in s and b in s:
            return f"Mutual exclusion: {a} with {b}"
    return None

def check_prereqs_satisfied(course_id: str, schedule_before_index: int, terms: List[TermPlan], prereqs: Dict[str, Prereq], student: StudentProfile = None) -> bool:
    pr = prereqs.get(course_id)
    if not pr or not pr.requires:
        return True
    taken: Set[str] = set()
    for i in range(0, schedule_before_index):
        taken.update(terms[i].courses)
    
    # Also include courses the student has credit for
    if student and student.courses_with_credit:
        taken.update(student.courses_with_credit)
    
    # Special: CHEM 11 can substitute for CHEM 06 in CHEM 51 and CHEM 52
    if course_id in ["CHEM 51", "CHEM 52"]:
        needed = []
        for r in pr.requires:
            if r == "CHEM 06":
                if ("CHEM 06" in taken) or ("CHEM 11" in taken):
                    continue
                else:
                    needed.append(r)
            elif r == "CHEM 11":
                if ("CHEM 11" in taken) or ("CHEM 06" in taken):
                    continue
                else:
                    needed.append(r)
            else:
                if r not in taken:
                    needed.append(r)
        return len(needed) == 0
    
    return all(r in taken for r in pr.requires)

# -----------------------------------------------------------------------------
# Planning Algorithm (Greedy placement with simple balancing)
# -----------------------------------------------------------------------------
def generate_plan(req: GeneratePlanRequest) -> Plan:
    student = req.student
    labels = term_sequence(student.entry_term, student.graduation_terms)
    terms: List[TermPlan] = []
    for label in labels:
        ttype = "on"
        if label in student.off_terms:
            ttype = "off"
        elif label in student.abroad_terms:
            ttype = "abroad"
        terms.append(TermPlan(label=label, type=ttype, courses=[], difficulty_sum=0, flags=[]))

    # Use request prereqs/rules if provided, otherwise fall back to global
    prereqs_dict = {p.course_id: p for p in req.prereqs} if req.prereqs else PREREQS
    rules_list = req.rules if req.rules else list(RULES.values())
    
    # Prepare data
    adjusted_required = _filter_and_expand_required(req.required_courses, prereqs_dict, student)
    order = topo_sort_courses(adjusted_required, prereqs_dict)
    
    # Prioritize CHEM 05 and CHEM 06 - move them to the front
    priority_courses = ["CHEM 05", "CHEM 06"]
    prioritized_order = [c for c in priority_courses if c in order]
    prioritized_order.extend([c for c in order if c not in priority_courses])
    order = prioritized_order
    
    required_set = set(adjusted_required)
    max_premed_per_term = 2
    
    # Count "on" terms to determine first 2 years (first 8 "on" terms)
    on_terms_indices = [i for i, t in enumerate(terms) if t.type == "on"]
    first_two_years_limit = on_terms_indices[7] if len(on_terms_indices) > 7 else len(terms) - 1

    # Place each course
    for cid in order:
        course = COURSES.get(cid)
        if not course:
            # Skip unknown courses instead of raising an error
            terms[-1].flags.append(f"Unknown course: {cid}")
            continue
        placed = False

        # Find earliest allowed index based on already placed prereqs
        earliest_allowed_idx = 0
        pr = prereqs_dict.get(cid)
        if pr and pr.requires:
            for r in pr.requires:
                latest_dep_idx = -1
                for tidx, t in enumerate(terms):
                    if r in t.courses:
                        latest_dep_idx = max(latest_dep_idx, tidx)
                    # Special case: CHEM 51 requires CHEM 06, but CHEM 11 can substitute
                    elif cid == "CHEM 51" and r == "CHEM 06":
                        if "CHEM 11" in t.courses:
                            latest_dep_idx = max(latest_dep_idx, tidx)
                if latest_dep_idx >= 0:
                    earliest_allowed_idx = max(earliest_allowed_idx, latest_dep_idx + 1)
        
        # For CHEM 05 and CHEM 06, ensure earliest_allowed_idx doesn't exceed first 2 years
        if cid in ["CHEM 05", "CHEM 06"]:
            earliest_allowed_idx = min(earliest_allowed_idx, first_two_years_limit)

        # Primary pass - favor single-course terms and spread difficulty
        # Calculate average difficulty per term to balance workload
        total_difficulty = sum(COURSES[c].difficulty for c in adjusted_required if c in COURSES)
        avg_difficulty_per_term = total_difficulty / max(1, len([t for t in terms if t.type == "on"]))
        
        # Score each potential term placement
        best_term_idx = -1
        best_score = float('inf')
        
        for idx, term in enumerate(terms[earliest_allowed_idx:], start=earliest_allowed_idx):
            # Only use "on" terms
            if term.type != "on":
                continue
            # Limit CHEM 05 and CHEM 06 to first 2 years (first 8 "on" terms)
            if cid in ["CHEM 05", "CHEM 06"] and idx > first_two_years_limit:
                continue
            # Check if course can be taken in this term
            if not can_take_in_term(course, term.label):
                continue
            # Check term restrictions
            if not is_term_allowed_for_classes(term.label, student.entry_term):
                continue
            premed_count = sum(1 for c in term.courses if c in required_set)
            if premed_count >= max_premed_per_term:
                continue
            if len(term.courses) >= 2:
                continue
            if not check_prereqs_satisfied(cid, idx, terms, prereqs_dict, student):
                continue
            prospective_difficulty = term.difficulty_sum + course.difficulty
            if req.difficulty_cap is not None and prospective_difficulty > req.difficulty_cap:
                continue
            ex = violates_mutual_exclusion(term.courses + [cid], rules_list)
            if ex:
                continue
            
            # Calculate placement score (lower is better)
            score = 0
            
            # Strong priority for CHEM 05 and CHEM 06 to be placed early
            if cid in ["CHEM 05", "CHEM 06"]:
                score -= 50  # Very strong preference for early placement
            
            # Prefer empty terms first, then single-course terms
            if len(term.courses) == 0:
                score -= 10  # Strong preference for empty terms
            elif len(term.courses) == 1:
                score -= 5   # Moderate preference for single-course terms
            else:
                score += 5   # Penalty for 2-course terms
            
            # Prefer earlier terms
            score += idx * 0.5
            
            # Balance difficulty
            difficulty_diff = abs(prospective_difficulty - avg_difficulty_per_term)
            score += difficulty_diff * 2.0
            
            if score < best_score:
                best_score = score
                best_term_idx = idx
        
        # Place in the best term
        if best_term_idx >= 0:
            term = terms[best_term_idx]
            term.courses.append(cid)
            term.difficulty_sum += course.difficulty
            placed = True

        # Fallback pass (more aggressive - allows stacking when needed)
        if not placed:
            # For 4-year plans, be more flexible in fallback
            for j in range(earliest_allowed_idx, len(terms)):
                term = terms[j]
                if term.type != "on":
                    continue
                # Limit CHEM 05 and CHEM 06 to first 2 years (first 8 "on" terms)
                if cid in ["CHEM 05", "CHEM 06"] and j > first_two_years_limit:
                    continue
                if not can_take_in_term(course, term.label):
                    continue
                if not is_term_allowed_for_classes(term.label, student.entry_term):
                    continue
                premed_count = sum(1 for c in term.courses if c in required_set)
                if premed_count >= max_premed_per_term:
                    continue
                if len(term.courses) >= 2:
                    continue
                if not check_prereqs_satisfied(cid, j, terms, prereqs_dict, student):
                    continue
                prospective_difficulty = term.difficulty_sum + course.difficulty
                if req.difficulty_cap is not None and prospective_difficulty > req.difficulty_cap:
                    continue
                ex = violates_mutual_exclusion(term.courses + [cid], rules_list)
                if ex:
                    continue
                term.courses.append(cid)
                term.difficulty_sum += course.difficulty
                term.flags.append("placed_with_risk:capacity_or_prereq")
                placed = True
                break

        if not placed:
            # Debug: Add more specific error message
            if cid == "BIOL 40":
                debug_info = []
                for j in range(earliest_allowed_idx, len(terms)):
                    term = terms[j]
                    if term.type != "on":
                        debug_info.append(f"Term {term.label} is {term.type}")
                        continue
                    if not can_take_in_term(course, term.label):
                        debug_info.append(f"Term {term.label} - course not offered")
                        continue
                    if not is_term_allowed_for_classes(term.label, student.entry_term):
                        debug_info.append(f"Term {term.label} - not allowed for classes")
                        continue
                    if not check_prereqs_satisfied(cid, j, terms, prereqs_dict, student):
                        debug_info.append(f"Term {term.label} - prereqs not satisfied")
                        continue
                    debug_info.append(f"Term {term.label} - should be valid!")
                # Add to errors list instead of flags
                terms[-1].flags.append(f"Missing: {cid}")
            else:
                terms[-1].flags.append(f"Missing: {cid}")

    # Post validations
    warnings: List[str] = []
    errors: List[str] = []

    for term in terms:
        ex = violates_mutual_exclusion(term.courses, rules_list)
        if ex:
            errors.append(f"{term.label}: {ex}")
    for idx, term in enumerate(terms):
        for cid in term.courses:
            if not check_prereqs_satisfied(cid, idx, terms, prereqs_dict, student):
                errors.append(f"{term.label}: {cid} missing prerequisite(s).")

    # Simple score: penalize errors and load variance
    load_variance = _term_load_variance(terms)
    score = max(0.0, 100.0 - 10.0 * len(errors) - 2.0 * len(warnings) - load_variance)

    return Plan(
        student_id=student.id,
        terms=terms,
        score=score,
        warnings=warnings,
        errors=errors,
        explanations=[],
        optional_courses=[c for c in (req.optional_courses or [])],
    )

def _term_load_variance(terms: List[TermPlan]) -> float:
    loads = [t.difficulty_sum for t in terms if t.type == "on"]
    if not loads:
        return 0.0
    avg = sum(loads) / len(loads)
    return sum((x - avg) ** 2 for x in loads) / len(loads)

# -----------------------------------------------------------------------------
# Validators
# -----------------------------------------------------------------------------
def validate_plan(req: ValidatePlanRequest) -> Plan:
    plan = req.plan
    warnings: List[str] = []
    errors: List[str] = []

    # Build from request or fall back to globals
    prereqs_dict = {p.course_id: p for p in req.prereqs} if req.prereqs else PREREQS
    rules_list = req.rules if req.rules else list(RULES.values())

    # Mutual exclusion per term
    for term in plan.terms:
        ex = violates_mutual_exclusion(term.courses, rules_list)
        if ex:
            errors.append(f"{term.label}: {ex}")

    # Prereqs: no student credits during validation
    for idx, term in enumerate(plan.terms):
        for cid in term.courses:
            if not check_prereqs_satisfied(cid, idx, plan.terms, prereqs_dict, student=None):
                errors.append(f"{term.label}: {cid} missing prerequisite(s).")

    load_variance = _term_load_variance(plan.terms)
    score = max(0.0, 100.0 - 10.0 * len(errors) - 2.0 * len(warnings) - load_variance)

    plan.warnings = warnings
    plan.errors = errors
    plan.score = score
    return plan


# -----------------------------------------------------------------------------
# FastAPI app and endpoints
# -----------------------------------------------------------------------------
 

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/test")
async def test():
    return {"message": "Backend is working!", "timestamp": "2024-01-01"}

@app.options("/{path:path}")
async def options_handler(path: str):
    return {"message": "OK"}

# Serve the frontend directly from the backend
@app.get("/")
async def serve_frontend():
    from fastapi.responses import Response
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dartmouth Pre-Health Planner</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 0; 
            background: #f5f5f5;
            min-height: 100vh;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
            overflow: hidden; 
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .header { 
            background: #00693e; 
            color: white; 
            padding: 40px; 
            text-align: center; 
            margin: 0; 
            position: relative;
            overflow: hidden;
        }
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="0.5" fill="white" opacity="0.1"/><circle cx="90" cy="40" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.3;
        }
        .header h1 { 
            margin: 0; 
            font-size: 3em; 
            font-weight: 700; 
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }
        .content { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 0; 
            min-height: 700px; 
        }
        .section { 
            padding: 40px; 
            border-right: 1px solid #e9ecef; 
            background: #fafbfc;
        }
        .section:last-child { 
            border-right: none; 
            background: #ffffff;
        }
        .section h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 25px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .course-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); 
            gap: 12px; 
            margin: 25px 0; 
        }
        .course-btn { 
            padding: 12px 8px; 
            border: 2px solid #e1e8ed; 
            background: white; 
            border-radius: 12px; 
            cursor: pointer; 
            font-size: 13px; 
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .course-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            transition: left 0.5s;
        }
        .course-btn:hover::before {
            left: 100%;
        }
        .course-btn:hover { 
            border-color: #00693e; 
            background: #f0f8f4;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 105, 62, 0.15);
        }
        .course-btn.selected { 
            background: #00693e; 
            color: white; 
            border-color: #00693e;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 105, 62, 0.3);
        }
        .btn { 
            padding: 12px 24px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 14px; 
            font-weight: 600;
            margin: 8px; 
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .btn-primary { 
            background: #00693e; 
            color: white; 
        }
        .btn-success { 
            background: #00693e; 
            color: white; 
        }
        .btn-secondary { 
            background: #6c757d; 
            color: white; 
        }
        .form-group { 
            margin-bottom: 20px; 
        }
        .form-group label { 
            display: block; 
            margin-bottom: 8px; 
            font-weight: 600; 
            color: #2c3e50;
            font-size: 14px;
        }
        .form-group input { 
            width: 100%; 
            padding: 12px 16px; 
            border: 2px solid #e1e8ed; 
            border-radius: 12px; 
            font-size: 14px;
            transition: all 0.3s ease;
            background: white;
        }
        .form-group input:focus {
            outline: none;
            border-color: #00693e;
            box-shadow: 0 0 0 3px rgba(0, 105, 62, 0.1);
        }
        .selected-courses { 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 15px; 
            margin: 20px 0; 
            border: 1px solid #dee2e6;
        }
        .selected-courses h3 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 1.3em;
        }
        .term-card { 
            background: #f8f9fa; 
            border: 1px solid #e9ecef; 
            border-radius: 4px; 
            padding: 15px; 
            margin: 10px 0; 
        }
        .success { 
            background: #d4edda; 
            padding: 15px; 
            border-radius: 12px; 
            border: 1px solid #c3e6cb; 
        }
        .error { 
            background: #f8d7da; 
            padding: 15px; 
            border-radius: 12px; 
            border: 1px solid #f5c6cb; 
        }
        .gap-year-checkbox {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 15px;
            background: #e8f5e8;
            border-radius: 12px;
            border: 2px solid #00693e;
            margin: 15px 0;
        }
        .gap-year-checkbox input[type="checkbox"] {
            transform: scale(1.5);
            accent-color: #00693e;
        }
        .gap-year-checkbox span {
            font-weight: 600;
            color: #00693e;
            font-size: 16px;
        }
        .gap-year-checkbox small {
            color: #666;
            font-size: 12px;
            margin-top: 5px;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
                <h1>Dartmouth Pre-Health Planner</h1>
            <p>Select your courses and generate a personalized schedule</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📚 Available Courses</h2>
                <p>Click on course buttons to select them:</p>
                
                <div style="margin: 20px 0;">
                    <button class="btn btn-secondary" id="selectAllBtn">Select All</button>
                    <button class="btn btn-secondary" id="selectPremedBtn">Select Pre-Med</button>
                    <button class="btn btn-secondary" id="clearAllBtn">Clear All</button>
                </div>
                
                <div class="course-grid" id="courseGrid">
                    <button class="course-btn" data-course="CHEM 05">CHEM 05</button>
                    <button class="course-btn" data-course="CHEM 06">CHEM 06</button>
                    <button class="course-btn" data-course="CHEM 11">CHEM 11</button>
                    <button class="course-btn" data-course="CHEM 51">CHEM 51</button>
                    <button class="course-btn" data-course="CHEM 52">CHEM 52</button>
                    <button class="course-btn" data-course="CHEM 41">CHEM 41</button>
                    <button class="course-btn" data-course="BIOL 12">BIOL 12</button>
                    <button class="course-btn" data-course="BIOL 13">BIOL 13</button>
                    <button class="course-btn" data-course="BIOL 14">BIOL 14</button>
                    <button class="course-btn" data-course="BIOL 40">BIOL 40</button>
                    <button class="course-btn" data-course="PHYS 03">PHYS 03</button>
                    <button class="course-btn" data-course="PHYS 04">PHYS 04</button>
                    <button class="course-btn" data-course="MATH 10">MATH 10</button>
                    <button class="course-btn" data-course="PSYC 01">PSYC 01</button>
                    <button class="course-btn" data-course="SOCY 01">SOCY 01</button>
                </div>
                
                
                
                <h2>👤 Student Profile</h2>
                <div class="form-group">
                    <label>Off Terms (comma-separated):</label>
                    <input type="text" id="offTerms" placeholder="25W,26S">
                </div>
                <div class="gap-year-checkbox">
                    <input type="checkbox" id="gapYear" checked>
                    <div>
                        <span>Include Gap Year (4-year plan)</span>
                        <small>Uncheck for 3-year accelerated plan</small>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📅 Generated Plan</h2>
                <button class="btn btn-success" id="generateBtn">
                    Generate Plan (0 courses)
                </button>
                
                <div id="planOutput">
                    <p>Select courses and click "Generate Plan" to see your schedule</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedCourses = [];
        
        window.toggleCourse = function(courseId) {
            const index = selectedCourses.indexOf(courseId);
            if (index > -1) {
                selectedCourses.splice(index, 1);
            } else {
                selectedCourses.push(courseId);
            }
            updateUI();
        };
        
        window.selectAll = function() {
            const buttons = document.querySelectorAll('.course-btn');
            selectedCourses = [];
            buttons.forEach(btn => {
                const courseId = btn.textContent.trim();
                selectedCourses.push(courseId);
            });
            updateUI();
        };
        
        window.selectPremed = function() {
            const premedCourses = ['CHEM 05', 'CHEM 06', 'CHEM 51', 'CHEM 52', 'BIOL 12', 'BIOL 13', 'BIOL 14', 'BIOL 40', 'PHYS 03', 'PHYS 04'];
            selectedCourses = premedCourses;
            updateUI();
        };
        
        window.clearAll = function() {
            selectedCourses = [];
            updateUI();
        };
        
        function updateUI() {
            try {
                // Update course buttons
                document.querySelectorAll('.course-btn').forEach(btn => {
                    const courseId = btn.textContent.trim();
                    if (selectedCourses.includes(courseId)) {
                        btn.classList.add('selected');
                    } else {
                        btn.classList.remove('selected');
                    }
                });
                
                // Update generate button
                const generateBtn = document.getElementById('generateBtn');
                if (generateBtn) {
                    generateBtn.textContent = `Generate Plan (${selectedCourses.length} courses)`;
                    generateBtn.disabled = selectedCourses.length === 0;
                }
            } catch (error) {
                console.error('Error in updateUI:', error);
            }
        }
        
        window.generatePlan = async function() {
            if (selectedCourses.length === 0) {
                alert('Please select at least one course');
                return;
            }
            
            const planOutput = document.getElementById('planOutput');
            planOutput.innerHTML = '<p>Generating plan...</p>';
            
            try {
                // First seed the backend
                await fetch('/seed/minimal', { method: 'POST' });
                
                const gapYear = document.getElementById('gapYear').checked;
                const graduationTerms = gapYear ? 16 : 12; // 4 years vs 3 years
                
                // If CHEM 11 is selected, add it to eligible courses for substitution logic
                const eligibleCourses = selectedCourses.includes('CHEM 11') ? ['CHEM 11'] : [];
                
                // Get off terms from input and add default off terms
                const inputOffTerms = document.getElementById('offTerms').value.split(',').map(t => t.trim()).filter(t => t);
                const defaultOffTerms = ['26X', '28X']; // Sophomore summer and senior summer are off by default
                const allOffTerms = [...new Set([...inputOffTerms, ...defaultOffTerms])]; // Remove duplicates
                
                const studentData = {
                    id: "student",
                    entry_term: "25F",
                    graduation_terms: graduationTerms,
                    max_courses_per_term: 2,
                    off_terms: allOffTerms,
                    abroad_terms: [],
                    incoming_credits: {},
                    courses_with_credit: [],
                    eligible_courses: eligibleCourses,
                    preferences: {}
                };
                
                const body = {
                    student: studentData,
                    required_courses: selectedCourses,
                    optional_courses: ['PSYC 01', 'SOCY 01'],
                    prereqs: [],
                    rules: [],
                    difficulty_cap: 7,
                };
                
                const response = await fetch('/plans/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }
                
                const plan = await response.json();
                displayPlan(plan);
                
            } catch (error) {
                planOutput.innerHTML = `<div class="error">
                    <h4>❌ Error generating plan:</h4>
                    <p><strong>Error:</strong> ${error.message}</p>
                </div>`;
            }
        };
        
        // Global variable to store the current plan
        let currentPlan = null;
        
        function displayPlan(plan) {
            currentPlan = plan; // Store the plan globally
            const planOutput = document.getElementById('planOutput');
            
            const gapYear = document.getElementById('gapYear').checked;
            const planType = gapYear ? '4-Year Plan (with Gap Year)' : '3-Year Accelerated Plan';
            let html = `<h3>📅 Your Course Schedule Chart - ${planType}</h3>`;
            html += '<p style="color: #666; font-size: 14px; margin-bottom: 20px;">💡 <strong>Drag and drop courses between terms to customize your schedule!</strong></p>';
            
            // Create a grid chart
            html += '<div id="planGrid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0;">';
            
            plan.terms.forEach((term, termIndex) => {
                const hasCourses = term.courses.length > 0;
                const termColor = hasCourses ? '#e3f2fd' : '#f5f5f5';
                const borderColor = hasCourses ? '#2196f3' : '#ccc';
                
                html += `<div id="term-${termIndex}" class="term-container" style="
                    background: ${termColor};
                    border: 2px solid ${borderColor};
                    border-radius: 8px;
                    padding: 15px;
                    text-align: center;
                    min-height: 120px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    transition: all 0.3s ease;
                " ondrop="drop(event, ${termIndex})" ondragover="allowDrop(event)">`;
                
                // Term header
                html += `<div style="font-weight: bold; color: #333; margin-bottom: 10px;">`;
                html += `<div style="font-size: 14px;">${term.label}</div>`;
                html += `<div style="font-size: 12px; color: #666;">${term.type}</div>`;
                html += '</div>';
                
                // Courses container
                html += `<div id="courses-${termIndex}" class="courses-container" style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center; min-height: 60px;">`;
                
                if (term.courses.length > 0) {
                    term.courses.forEach(course => {
                        html += `<div class="course-item" draggable="true" ondragstart="drag(event, '${course}')" ondragend="dragEnd(event)" style="
                            background: #00693e;
                            color: white;
                            padding: 4px 8px;
                            margin: 2px 0;
                            border-radius: 12px;
                            font-size: 11px;
                            font-weight: 500;
                            cursor: move;
                            transition: all 0.2s ease;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                        " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">${course}</div>`;
                    });
                } else {
                    html += '<div style="color: #999; font-size: 12px;">Drop courses here</div>';
                }
                
                html += '</div>'; // Close courses container
                
                // Flags (if any)
                if (term.flags.length > 0) {
                    html += '<div style="margin-top: 8px; font-size: 10px; color: #666;">';
                    term.flags.forEach(flag => {
                        html += `<div style="background: #fff3cd; padding: 2px 4px; border-radius: 3px; margin: 1px 0;">${flag}</div>`;
                    });
                    html += '</div>';
                }
                
                html += '</div>'; // Close term container
            });
            
            html += '</div>'; // Close grid

            // Add course palette for drag and drop of unplaced courses
            // Compute placed courses
            const placedCourses = new Set();
            plan.terms.forEach(t => t.courses.forEach(c => placedCourses.add(c)));

            // All available courses (pre-med focused + optional)
            const allCourses = [
                'CHEM 05','CHEM 06','CHEM 11','CHEM 51','CHEM 52','CHEM 41',
                'BIOL 12','BIOL 13','BIOL 14','BIOL 40',
                'PHYS 03','PHYS 04','MATH 10',
                'PSYC 01','SOCY 01'
            ];

            const availableCourses = allCourses.filter(c => !placedCourses.has(c));

            html += '<div id="paletteContainer" style="margin-top: 20px; padding: 16px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px;">';
            html += '<h4 style="margin: 0 0 10px 0; color: #2c3e50;">📚 Available Courses (Drag into calendar or drag courses here to remove)</h4>';
            html += '<div id="coursePalette" style="display: flex; flex-wrap: wrap; gap: 8px; min-height: 50px;" ondrop="dropOnPalette(event)" ondragover="allowDropOnPalette(event)">';
            if (availableCourses.length > 0) {
                availableCourses.forEach(course => {
                    html += `<div class="palette-course" draggable="true" ondragstart="drag(event, '${course}')" ondragend="dragEnd(event)" style="
                        background: #00693e; color: white; padding: 6px 10px; border-radius: 14px; font-size: 12px; font-weight: 600; cursor: move; border: 2px solid transparent; display: flex; justify-content: center; align-items: center;">
                        ${course}
                    </div>`;
                });
            } else {
                html += '<div style="color:#666; font-size:12px; font-style:italic;">All courses are placed.</div>';
            }
            html += '</div>';
            html += '<div style="color:#666; font-size:12px; margin-top:6px;">Tip: Drag courses from calendar into this area to remove them, or drag from here into calendar to add them.</div>';
            html += '</div>';

            // Collect missing courses from flags and remove ones already placed
            let missingCourses = [];
            plan.terms.forEach(term => {
                term.flags.forEach(flag => {
                    if (flag.startsWith('Missing:')) {
                        missingCourses.push(flag.replace('Missing: ', ''));
                    }
                });
            });
            const placedNow = new Set();
            plan.terms.forEach(t => t.courses.forEach(c => placedNow.add(c)));
            missingCourses = missingCourses.filter(c => !placedNow.has(c));
            
            // Show missing courses at the bottom
            if (missingCourses.length > 0) {
                html += '<div style="margin-top: 20px; padding: 15px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px;">';
                html += '<h4 style="color: #721c24; margin: 0 0 10px 0;">❌ Missing Courses:</h4>';
                html += '<p style="color: #721c24; margin: 0; font-weight: 500;">' + missingCourses.join(', ') + '</p>';
                html += '</div>';
            }
            
            // Errors and warnings
            if (plan.errors.length > 0) {
                html += '<div class="error">';
                html += '<h4>❌ Errors:</h4><ul>';
                plan.errors.forEach(error => {
                    html += `<li>${error}</li>`;
                });
                html += '</ul></div>';
            }
            
            if (plan.warnings.length > 0) {
                html += '<div style="background: #fff3cd; padding: 15px; margin: 15px 0; border-radius: 4px; border: 1px solid #ffeaa7;">';
                html += '<h4>⚠️ Warnings:</h4><ul>';
                plan.warnings.forEach(warning => {
                    html += `<li>${warning}</li>`;
                });
                html += '</ul></div>';
            }
            
            
            planOutput.innerHTML = html;
        }
        
        // Drag and Drop Functions
        function allowDrop(ev) {
            ev.preventDefault();
            
            // Get the course being dragged
            const draggedCourse = ev.dataTransfer.getData("text");
            const targetTermIndex = parseInt(ev.currentTarget.id.split('-')[1]);
            
            // Check if the drop is valid
            const isValid = validateCourseDrop(draggedCourse, targetTermIndex);
            
            if (isValid) {
                ev.currentTarget.style.borderColor = '#00693e';
                ev.currentTarget.style.backgroundColor = '#f0f8f4';
            } else {
                ev.currentTarget.style.borderColor = '#dc3545';
                ev.currentTarget.style.backgroundColor = '#f8d7da';
            }
        }
        
        function drag(ev, courseName) {
            ev.dataTransfer.setData("text", courseName);
            ev.dataTransfer.effectAllowed = "move";
            draggedCourseName = courseName;
        }
        
        function dragEnd(ev) {
            // Reset palette styling when drag ends
            const palette = document.getElementById('coursePalette');
            if (palette) {
                palette.style.backgroundColor = '';
                palette.style.borderColor = '';
                palette.style.borderWidth = '';
                palette.style.borderStyle = '';
            }
            draggedCourseName = null;
        }
        
        function drop(ev, targetTermIndex) {
            ev.preventDefault();
            const courseName = ev.dataTransfer.getData("text");
            
            // Reset styling
            ev.currentTarget.style.borderColor = '';
            ev.currentTarget.style.backgroundColor = '';
            
            // Validate the drop before proceeding
            if (!validateCourseDrop(courseName, targetTermIndex)) {
                showNotification(`Cannot move ${courseName} to ${currentPlan.terms[targetTermIndex].label} - violates constraints!`, 'error');
                return;
            }
            
            // Find the source term and remove the course
            let sourceTermIndex = -1;
            for (let i = 0; i < currentPlan.terms.length; i++) {
                const courseIndex = currentPlan.terms[i].courses.indexOf(courseName);
                if (courseIndex !== -1) {
                    sourceTermIndex = i;
                    currentPlan.terms[i].courses.splice(courseIndex, 1);
                    break;
                }
            }
            
            // Add the course to the target term
            if (sourceTermIndex !== targetTermIndex) {
                currentPlan.terms[targetTermIndex].courses.push(courseName);
                
                // Update the visual display
                updatePlanDisplay();
                
                // Show success message
                showNotification(`Moved ${courseName} to ${currentPlan.terms[targetTermIndex].label}`, 'success');
            }
        }
        
        // Global variable to track what's being dragged
        let draggedCourseName = null;
        
        function allowDropOnPalette(ev) {
            ev.preventDefault();
            // Highlight the palette area when dragging
            ev.currentTarget.style.backgroundColor = '#e8f5e8';
            ev.currentTarget.style.borderColor = '#00693e';
            ev.currentTarget.style.borderWidth = '2px';
            ev.currentTarget.style.borderStyle = 'dashed';
        }
        
        function dropOnPalette(ev) {
            ev.preventDefault();
            const courseName = ev.dataTransfer.getData("text");
            
            // Reset styling
            ev.currentTarget.style.backgroundColor = '';
            ev.currentTarget.style.borderColor = '';
            ev.currentTarget.style.borderWidth = '';
            ev.currentTarget.style.borderStyle = '';
            
            // Find and remove the course from the calendar
            let removed = false;
            for (let i = 0; i < currentPlan.terms.length; i++) {
                const courseIndex = currentPlan.terms[i].courses.indexOf(courseName);
                if (courseIndex !== -1) {
                    currentPlan.terms[i].courses.splice(courseIndex, 1);
                    removed = true;
                    break;
                }
            }
            
            if (removed) {
                // Update the visual display (this will add the course back to the palette)
                updatePlanDisplay();
            }
        }
        
        function updatePlanDisplay() {
            if (!currentPlan) return;
            
            // Update the visual display without regenerating the entire plan
            currentPlan.terms.forEach((term, termIndex) => {
                const coursesContainer = document.getElementById(`courses-${termIndex}`);
                if (coursesContainer) {
                    coursesContainer.innerHTML = '';
                    
                    if (term.courses.length > 0) {
                        term.courses.forEach(course => {
                            const courseElement = document.createElement('div');
                            courseElement.className = 'course-item';
                            courseElement.draggable = true;
                            courseElement.ondragstart = (e) => drag(e, course);
                            courseElement.ondragend = dragEnd;
                            courseElement.style.cssText = `
                                background: #00693e;
                                color: white;
                                padding: 4px 8px;
                                margin: 2px 0;
                                border-radius: 12px;
                                font-size: 11px;
                                font-weight: 500;
                                cursor: move;
                                transition: all 0.2s ease;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                            `;
                            courseElement.onmouseover = () => courseElement.style.transform = 'scale(1.05)';
                            courseElement.onmouseout = () => courseElement.style.transform = 'scale(1)';
                            courseElement.textContent = course;
                            coursesContainer.appendChild(courseElement);
                        });
                    } else {
                        coursesContainer.innerHTML = '<div style="color: #999; font-size: 12px;">Drop courses here</div>';
                    }
                }
                
                // Update term styling based on whether it has courses
                const termContainer = document.getElementById(`term-${termIndex}`);
                if (termContainer) {
                    const hasCourses = term.courses.length > 0;
                    termContainer.style.backgroundColor = hasCourses ? '#e3f2fd' : '#f5f5f5';
                    termContainer.style.borderColor = hasCourses ? '#2196f3' : '#ccc';
                }
            });

            // Rebuild palette based on placed courses
            const placed = new Set();
            currentPlan.terms.forEach(t => t.courses.forEach(c => placed.add(c)));
            
            const allCourses = [
                'CHEM 05','CHEM 06','CHEM 11','CHEM 51','CHEM 52','CHEM 41',
                'BIOL 12','BIOL 13','BIOL 14','BIOL 40',
                'PHYS 03','PHYS 04','MATH 10',
                'PSYC 01','SOCY 01'
            ];
            
            const availableCourses = allCourses.filter(c => !placed.has(c));
            const palette = document.getElementById('coursePalette');
            
            if (palette) {
                // Clear existing palette items
                palette.innerHTML = '';
                
                // Add available courses to palette
                if (availableCourses.length > 0) {
                    availableCourses.forEach(course => {
                        const courseElement = document.createElement('div');
                        courseElement.className = 'palette-course';
                        courseElement.draggable = true;
                        courseElement.ondragstart = (e) => drag(e, course);
                        courseElement.ondragend = dragEnd;
                        courseElement.style.cssText = 'background: #00693e; color: white; padding: 6px 10px; border-radius: 14px; font-size: 12px; font-weight: 600; cursor: move; border: 2px solid transparent; display: flex; justify-content: center; align-items: center;';
                        courseElement.textContent = course;
                        palette.appendChild(courseElement);
                    });
                } else {
                    palette.innerHTML = '<div style="color:#666; font-size:12px; font-style:italic;">All courses are placed.</div>';
                }
            }
        }
        
        function validateCourseDrop(courseName, targetTermIndex) {
            if (!currentPlan) return false;
            
            const targetTerm = currentPlan.terms[targetTermIndex];
            
            // Check if term is "on" (active)
            if (targetTerm.type !== "on") {
                return false;
            }
            
            // Check if course is offered in this term
            if (!isCourseOfferedInTerm(courseName, targetTerm.label)) {
                return false;
            }
            
            // Check max courses per term (assuming 2 for now)
            const maxCoursesPerTerm = 2;
            if (targetTerm.courses.length >= maxCoursesPerTerm) {
                return false;
            }
            
            // Check prerequisites
            if (!checkPrerequisites(courseName, targetTermIndex)) {
                return false;
            }
            
            // Check mutual exclusions
            if (hasMutualExclusion(courseName, targetTerm.courses)) {
                return false;
            }
            
            // Check if course is already in target term
            if (targetTerm.courses.includes(courseName)) {
                return false;
            }
            
            return true;
        }
        
        function checkPrerequisites(courseName, targetTermIndex) {
            // Define prerequisites with CHEM 11 substitution logic
            const prerequisites = {
                'CHEM 06': ['CHEM 05'],
                'CHEM 51': ['CHEM 06'], // Can be satisfied by CHEM 06 OR CHEM 11
                'CHEM 52': ['CHEM 51'], // Requires CHEM 51 (which can use CHEM 11)
                'CHEM 41': ['CHEM 52'],
                'BIOL 40': ['BIOL 12', 'CHEM 52'],
                'PHYS 04': ['PHYS 03']
            };
            
            // Find the current location of the course (if it's already placed)
            let currentTermIndex = -1;
            for (let i = 0; i < currentPlan.terms.length; i++) {
                if (currentPlan.terms[i].courses.includes(courseName)) {
                    currentTermIndex = i;
                    break;
                }
            }
            
            const prereqs = prerequisites[courseName];
            if (!prereqs) {
                // No prerequisites for this course, but check if moving it would violate prerequisites of courses that come after
                return checkIfMovingViolatesLaterPrereqs(courseName, targetTermIndex, currentTermIndex);
            }
            
            // Collect all courses placed before target term (excluding the course itself if it's already there)
            const coursesBefore = [];
            for (let i = 0; i < targetTermIndex; i++) {
                const termCourses = [...currentPlan.terms[i].courses];
                // If the course is currently in this term, exclude it from the prerequisite check
                if (i === currentTermIndex) {
                    const index = termCourses.indexOf(courseName);
                    if (index !== -1) {
                        termCourses.splice(index, 1);
                    }
                }
                coursesBefore.push(...termCourses);
            }
            
            // Check if all prerequisites are satisfied
            for (const prereq of prereqs) {
                let satisfied = false;
                
                // Special case: CHEM 51 can use CHEM 06 OR CHEM 11
                if (courseName === 'CHEM 51' && prereq === 'CHEM 06') {
                    if (coursesBefore.includes('CHEM 06') || coursesBefore.includes('CHEM 11')) {
                        satisfied = true;
                    }
                } else {
                    // Regular prerequisite check
                    if (coursesBefore.includes(prereq)) {
                        satisfied = true;
                    }
                }
                
                if (!satisfied) {
                    return false; // At least one prerequisite not satisfied
                }
            }
            
            // Also check if moving this course would violate prerequisites of courses that come after
            return checkIfMovingViolatesLaterPrereqs(courseName, targetTermIndex, currentTermIndex);
        }
        
        function checkIfMovingViolatesLaterPrereqs(courseName, targetTermIndex, currentTermIndex) {
            // Check if any course placed in targetTermIndex or AFTER requires courseName as a prerequisite
            const prerequisites = {
                'CHEM 06': ['CHEM 05'],
                'CHEM 51': ['CHEM 06'], // Can be satisfied by CHEM 06 OR CHEM 11
                'CHEM 52': ['CHEM 51'],
                'CHEM 41': ['CHEM 52'],
                'BIOL 40': ['BIOL 12', 'CHEM 52'],
                'PHYS 04': ['PHYS 03']
            };
            
            // Check all terms from targetTermIndex onwards
            for (let i = targetTermIndex; i < currentPlan.terms.length; i++) {
                const term = currentPlan.terms[i];
                for (const courseInTerm of term.courses) {
                    // Skip if this is the course we're moving (unless it's in a different term)
                    if (courseInTerm === courseName && i === currentTermIndex) {
                        continue;
                    }
                    
                    const prereqs = prerequisites[courseInTerm];
                    if (!prereqs) continue;
                    
                    // Check if this course requires courseName as a prerequisite
                    for (const prereq of prereqs) {
                        // Special case: CHEM 51 requires CHEM 06, but CHEM 11 can substitute
                        if (courseInTerm === 'CHEM 51' && prereq === 'CHEM 06') {
                            // If we're moving CHEM 06 or CHEM 11 to the same term or after CHEM 51, that's invalid
                            // (unless CHEM 51 already has CHEM 06 or CHEM 11 before it)
                            if (courseName === 'CHEM 06' || courseName === 'CHEM 11') {
                                // Check if CHEM 51 has its prerequisite satisfied in terms before it
                                // (excluding the course we're moving from its current location)
                                let hasPrereq = false;
                                for (let j = 0; j < i; j++) {
                                    const earlierTerm = currentPlan.terms[j];
                                    const termCourses = [...earlierTerm.courses];
                                    // Exclude the course we're moving if it's in this term
                                    if (j === currentTermIndex) {
                                        const index = termCourses.indexOf(courseName);
                                        if (index !== -1) {
                                            termCourses.splice(index, 1);
                                        }
                                    }
                                    if (termCourses.includes('CHEM 06') || termCourses.includes('CHEM 11')) {
                                        hasPrereq = true;
                                        break;
                                    }
                                }
                                // If CHEM 51 doesn't have its prerequisite yet, and we're moving CHEM 06/11 to same or later term, that's invalid
                                if (!hasPrereq) {
                                    return false;
                                }
                            }
                        } else if (prereq === courseName) {
                            // This course requires courseName, so courseName must come before it
                            // If we're moving courseName to the same term or after this course, that's invalid
                            return false;
                        }
                    }
                }
            }
            
            return true;
        }
        
        function hasMutualExclusion(courseName, existingCourses) {
            // Define mutual exclusions
            const mutualExclusions = {
                'CHEM 51': ['CHEM 52'],
                'CHEM 52': ['CHEM 51']
            };
            
            const exclusions = mutualExclusions[courseName];
            if (!exclusions) return false;
            
            return exclusions.some(excluded => existingCourses.includes(excluded));
        }
        
        function isCourseOfferedInTerm(courseName, termLabel) {
            // Extract season from term label (e.g., "25F" -> "F", "26W" -> "W")
            const season = termLabel.slice(-1);
            
            // Define which courses are offered in which terms
            const courseOfferings = {
                'CHEM 05': ['F', 'W'],
                'CHEM 06': ['W', 'S'],
                'CHEM 11': ['F', 'S'],
                'CHEM 51': ['F', 'W', 'S'],
                'CHEM 52': ['W', 'S', 'X'],
                'CHEM 41': ['S'],
                'BIOL 12': ['F', 'X'],
                'BIOL 13': ['W', 'S'],
                'BIOL 14': ['W', 'X'],
                'BIOL 40': ['F', 'W'],
                'PHYS 03': ['F', 'W', 'S'],
                'PHYS 04': ['W', 'S', 'X'],
                'MATH 10': ['F', 'W', 'S'],
                'PSYC 01': ['F', 'W', 'S'],
                'SOCY 01': ['F', 'W', 'S']
            };
            
            const offeredTerms = courseOfferings[courseName];
            if (!offeredTerms) return true; // If not specified, assume always offered
            
            return offeredTerms.includes(season);
        }
        
        function showNotification(message, type = 'info') {
            // Create notification element
            const notification = document.createElement('div');
            const bgColor = type === 'success' ? '#d4edda' : type === 'error' ? '#f8d7da' : '#d1ecf1';
            const textColor = type === 'success' ? '#155724' : type === 'error' ? '#721c24' : '#0c5460';
            const borderColor = type === 'success' ? '#c3e6cb' : type === 'error' ? '#f5c6cb' : '#bee5eb';
            
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: ${bgColor};
                color: ${textColor};
                border: 1px solid ${borderColor};
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 14px;
                font-weight: 500;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 1000;
                animation: slideIn 0.3s ease;
            `;
            notification.textContent = message;
            
            // Add animation styles
            const style = document.createElement('style');
            style.textContent = `
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOut {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
            
            document.body.appendChild(notification);
            
            // Remove notification after 3 seconds
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }, 3000);
        }
        
        // Initialize UI when DOM is ready
        function initializePage() {
            updateUI();
            
            // Attach event listeners to course buttons
            document.querySelectorAll('.course-btn').forEach(btn => {
                const courseId = btn.getAttribute('data-course') || btn.textContent.trim();
                btn.addEventListener('click', () => toggleCourse(courseId));
            });
            
            // Attach event listeners to action buttons
            const selectAllBtn = document.getElementById('selectAllBtn');
            if (selectAllBtn) {
                selectAllBtn.addEventListener('click', selectAll);
            }
            
            const selectPremedBtn = document.getElementById('selectPremedBtn');
            if (selectPremedBtn) {
                selectPremedBtn.addEventListener('click', selectPremed);
            }
            
            const clearAllBtn = document.getElementById('clearAllBtn');
            if (clearAllBtn) {
                clearAllBtn.addEventListener('click', clearAll);
            }
            
            const generateBtn = document.getElementById('generateBtn');
            if (generateBtn) {
                generateBtn.addEventListener('click', generatePlan);
            }
        }
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializePage);
        } else {
            initializePage();
        }
    </script>
</body>
</html>
    """
    return Response(content=html_content, media_type="text/html", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })

@app.get("/courses", response_model=List[Course])
async def list_courses():
    return list(COURSES.values())

@app.post("/courses", response_model=Course)
async def create_course(course: Course):
    if course.id in COURSES:
        raise HTTPException(status_code=400, detail="Course already exists")
    COURSES[course.id] = course
    return course

@app.get("/rules", response_model=List[Rule])
async def list_rules():
    return list(RULES.values())

@app.post("/profiles", response_model=StudentProfile)
async def create_profile(profile: StudentProfile):
    PROFILES[profile.id] = profile
    return profile

@app.post("/plans/generate", response_model=Plan)
async def api_generate_plan(body: GeneratePlanRequest):
    # Use the prereqs and rules from the request, or fall back to global ones
    prereqs_dict = {p.course_id: p for p in body.prereqs} if body.prereqs else PREREQS
    rules_list = body.rules if body.rules else list(RULES.values())
    
    # Create a new request with the proper prereqs and rules
    updated_body = GeneratePlanRequest(
        student=body.student,
        required_courses=body.required_courses,
        optional_courses=body.optional_courses,
        prereqs=list(prereqs_dict.values()),
        rules=rules_list,
        difficulty_cap=body.difficulty_cap
    )
    return generate_plan(updated_body)

@app.post("/plans/validate", response_model=Plan)
async def api_validate_plan(body: ValidatePlanRequest):
    for p in body.prereqs:
        PREREQS[p.course_id] = p
    for r in body.rules:
        RULES[r.id] = r
    return validate_plan(body)

# -----------------------------------------------------------------------------
# Minimal Seed Loader — ~15 courses + basic rules for demos
# -----------------------------------------------------------------------------
class SeedResult(BaseModel):
    courses: int
    rules: int

@app.post("/seed/minimal", response_model=SeedResult)
async def seed_minimal():
    COURSES.clear(); RULES.clear(); PREREQS.clear()
    seed_courses = [
        # CHEM
        Course(id="CHEM 05", title="General Chemistry I", dept="CHEM", difficulty=3, tags=["lab"], typical_terms=["F","W"]),
        Course(id="CHEM 06", title="General Chemistry II", dept="CHEM", difficulty=3, tags=["lab"], typical_terms=["W","S"]),
        Course(id="CHEM 11", title="Advanced General Chemistry", dept="CHEM", difficulty=4, tags=["lab"], typical_terms=["F","S"]),
        Course(id="CHEM 51", title="Organic Chemistry I", dept="CHEM", difficulty=4, tags=["lab"], typical_terms=["F","W","S"]),
        Course(id="CHEM 52", title="Organic Chemistry II", dept="CHEM", difficulty=4, tags=["lab"], typical_terms=["W","S","X"]),
        Course(id="CHEM 41", title="Biochemistry (Chem)", dept="CHEM", difficulty=4, tags=["lab"], typical_terms=["S"]),
        # BIOL
        Course(id="BIOL 12", title="Cell Biology", dept="BIOL", difficulty=4, tags=["lab"], typical_terms=["F","X"]),
        Course(id="BIOL 13", title="Genetics", dept="BIOL", difficulty=4, tags=["pset"], typical_terms=["W","S"]),
        Course(id="BIOL 14", title="Physiology", dept="BIOL", difficulty=4, tags=["lab"], typical_terms=["W","X"]),
        Course(id="BIOL 40", title="Biochemistry (Bio)", dept="BIOL", difficulty=4, tags=["lab"], typical_terms=["F","W"]),
        # PHYS
        Course(id="PHYS 03", title="Intro Physics I", dept="PHYS", difficulty=3, tags=["math"], typical_terms=["F","X"]),
        Course(id="PHYS 04", title="Intro Physics II", dept="PHYS", difficulty=3, tags=["math"], typical_terms=["W","S"]),
        # MATH & Social
        Course(id="MATH 10", title="Intro Statistics", dept="MATH", difficulty=2, tags=["stats"], typical_terms=["S"]),
        Course(id="PSYC 01", title="Intro Psychology", dept="PSYC", difficulty=2, tags=["social"], typical_terms=["F","S"]),
        Course(id="SOCY 01", title="Intro Sociology", dept="SOCY", difficulty=2, tags=["social"], typical_terms=["F","S"]),
    ]
    for c in seed_courses:
        COURSES[c.id] = c

    # Prereqs
    prereqs = [
        Prereq(course_id="CHEM 06", requires=["CHEM 05"]),
        Prereq(course_id="CHEM 51", requires=["CHEM 06"]),  # Orgo I after Gen Chem II
        Prereq(course_id="CHEM 52", requires=["CHEM 51"]),  # Orgo II after Orgo I
        Prereq(course_id="CHEM 41", requires=["CHEM 52"]),          # Biochem after Orgo II (Chem version)
        Prereq(course_id="BIOL 40", requires=["BIOL 12", "CHEM 52"]),# Biochem (Bio) after Bio + Orgo II
        Prereq(course_id="PHYS 04", requires=["PHYS 03"]),          # Physics II after I
    ]
    for p in prereqs:
        PREREQS[p.course_id] = p

    # Rules
    rules = [
        Rule(id="no_orgo_pair", type="mutual_exclusion", severity="error",
             message="Cannot take CHEM 51 and CHEM 52 in the same term",
             course_pair=("CHEM 51", "CHEM 52"))
    ]
    for r in rules:
        RULES[r.id] = r

    return SeedResult(courses=len(COURSES), rules=len(RULES))