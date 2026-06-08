# Xybrid Project Governance

This document describes how the Xybrid project is organized and how decisions
are made. It is intentionally lightweight and reflects how the project operates
today; it will evolve as the project and community grow.

Xybrid is free and open-source software released under the
[Apache License 2.0](LICENSE) and stewarded by Xybrid, Inc. Development happens
in the open on [GitHub](https://github.com/xybrid-ai/xybrid).

## Roles

- **Users** — anyone who uses Xybrid. Users are encouraged to ask questions,
  report bugs, and request features via [GitHub Issues](https://github.com/xybrid-ai/xybrid/issues)
  and [Discord](https://discord.gg/YhFHHkhbad).
- **Contributors** — anyone who submits a contribution (code, documentation,
  tests, triage, reviews). The contribution process is described in
  [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are made under the
  project's [Contributor License Agreement](CLA.md) and Apache-2.0 license.
- **Maintainers** — contributors with elevated access who are responsible for
  the long-term health of the project. The current maintainers are listed in
  [MAINTAINERS.md](MAINTAINERS.md).

## Decision-making

Day-to-day changes are proposed as pull requests and merged once they pass CI
and receive maintainer review (see [Change review](#change-review)). The project
operates by lazy consensus: a proposal may proceed if no maintainer objects.

Significant or potentially contentious changes (public API changes, new
dependencies with broad impact, changes to release or security processes)
should first be discussed in a GitHub Issue or draft PR so that maintainers and
the community can weigh in. When consensus cannot be reached, the maintainers
make the final decision.

## Change review

All changes to the primary branch (`master`) go through a pull request. Branch
protection enforces that:

- direct pushes to `master` are not allowed;
- required status checks (CI) must pass before merge;
- changes are reviewed before merge.

Maintainers may expedite low-risk changes where the project's review policy
permits. See [MAINTAINERS.md](MAINTAINERS.md) for who holds merge and release
authority.

## Becoming a maintainer

Maintainers are added by invitation from the existing maintainers, based on a
sustained track record of high-quality contributions and reviews and good
judgement about the project's direction. If you are interested in taking on more
responsibility, the best path is to contribute consistently and help review
others' work.

## Code of Conduct

Participation in the project is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). Maintainers are responsible for
enforcing it.

## Changes to this document

Changes to the project's governance are proposed via pull request and require
approval from the maintainers.
