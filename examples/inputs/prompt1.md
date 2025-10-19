Project: Minimal “Customer Feedback Inbox” for a SaaS dashboard

Context
- Our web app (React + Node/Express + Postgres) lacks a way to capture and triage user feedback.
- We want a lightweight MVP in 2–3 days.

Requirement
- Add a “Feedback” widget to the app that lets logged-in users submit:
  - Category (Bug, Feature, Other), short title (max 80 chars), description (max 500 chars), optional screenshot URL.
- Create a basic triage view for internal staff:
  - Filter by category and status (New, In Review, Closed), search by title, and change status.
- Store submissions in Postgres. Keep schema simple.
- Authentication: Only logged-in users can submit; only staff (role=admin) can access triage.
- Performance: first contentful interaction < 1s for the widget on broadband.
- Accessibility: form keyboard navigable; labels + aria-* where needed.

Constraints & Nice-to-haves
- Keep UI minimal; use existing design tokens.
- No third-party feedback vendors; ok to add 1 small NPM lib if justified.
- Log key events for analytics (submit, status change) via existing logger.

Success Criteria
- Staff can review and update status on desktop and mobile.
- At least 1 unit test per critical path; basic error states covered.
- Ship with a short README: setup, env vars, and how to run tests.
