//! The library half of `xybrid-cli`: what the `xybrid` binary shares with
//! its tests.
//!
//! Internal: an interface between the CLI and its tests, with no stability
//! promise. It is not published and not documented as an API; anything in
//! it can change with the CLI.
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`systemone`] | The SystemOne transport of `xybrid run --decision`: request parsing and response writing |

#[doc(hidden)]
pub mod systemone;
