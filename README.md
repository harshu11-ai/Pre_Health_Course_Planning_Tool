# Pre_Health_Course_Planning_Tool
# Pre‑Health Course Planning Tool

A modern web-based tool to help Dartmouth students (and others) plan their pre-health course schedules, balance workloads, and satisfy common medical school admissions requirements in either standard or accelerated tracks. The tool intelligently recommends a course-taking sequence while enforcing prerequisites, mutual exclusions, and typical offering terms. Users can interactively adjust their plan, visualize term-by-term loads, and get warnings about unmet prerequisites or schedule issues.

---

## Features

- **Automatic Course Sequencing:** Suggests an optimized course plan that respects prerequisites, maximum difficulty, and term-specific constraints.
- **Interactive UI:** Allows drag-and-drop rearrangement of courses between terms in a visual grid.
- **Flexible Plan Options:** Supports standard 4-year and accelerated 3-year+gap year schedules.
- **Real-Time Validations:** Instantly warns about invalid prerequisites, course conflicts, and scheduling constraints as you adjust your plan.
- **Customizable Requirements:** Add or remove courses, select all, select only pre-med essentials, or clear your plan.
- **Built-in Knowledge:** Includes logic for when courses are typically offered, prerequisite chains, and mutual exclusions.

---

## Usage

### 1. **Run the Backend**

Make sure you have Python 3.9+ and `fastapi` installed.

```
pip install fastapi uvicorn pydantic
uvicorn main:app --reload
```

This will start the backend server at [http://localhost:8000](http://localhost:8000).

### 2. **Open the Tool in Your Browser**

The tool includes a static frontend. By default, visit:

```
http://localhost:8000
```

Use the interactive controls to generate a plan, select/deselect courses, and visually arrange your schedule.

---

## How It Works

- **Course Requirements:** Built-in models incorporate Dartmouth’s typical pre-med track (general chemistry, organic chemistry, biology, physics, math, psychology, sociology, etc.).
- **Prereqs & Constraints:** Prerequisite logic ensures no course is scheduled before its requirements are met; mutual exclusion logic prevents taking conflicting courses together.
- **Term Awareness:** Respects which courses are offered in each academic term (Fall, Winter, Spring, Summer "X").
- **Smart Placement:** Optimizes to avoid overloading any term based on course difficulty, while prioritizing spreading out required science courses.
- **Gap Year Option:** Customizes the plan for students pursuing a gap year before applying to medical school.

---

## Development

### Project Structure

- `main.py` — FastAPI backend, plan generation logic, scheduling rules and endpoints.
- `static/` — The HTML/CSS/JavaScript frontend (served automatically by FastAPI's static files handler).
- `README.md` — This documentation.

### Tech & Dependencies

- **Backend:** FastAPI, Pydantic for models and validation.
- **Frontend:** Vanilla JavaScript with modern dynamic HTML, drag-and-drop, and UI feedback.
- **Host/Run:** Only Python needed to run locally; works cross‑platform.

---

## Example Screenshot

![Example course plan grid UI](https://user-images.githubusercontent.com/example/plan-grid.png)

---

## Roadmap / Future Work

- Allow user-defined electives and requirements
- Import/export plans
- Integration with student info (via authentication)
- Advisor workflow for approving/suggesting plans
- Mobile-friendly UI refinements

---

## License

This project is a prototype for educational use. Feel free to fork, adapt, or extend for non-commercial and non-clinical use.

---

## Authors

- Harshith Yallampalli
- Inspired by Dartmouth pre-health advising resources

---

