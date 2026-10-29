/*
 * A member of the fake engine that nothing references, like the dead members
 * of a real engine archive. It calls a function no library defines, so
 * libselftest.so links (under the toolchain's --no-undefined) only if
 * --gc-sections drops this member's .text. Built without function sections,
 * like fake_engine.c, that is a plain .text in both members: ld -r merges
 * the two unless --unique keeps them apart (bazel/zzz/isolate_engine.sh).
 */
extern void fake_engine_unresolved(void);

void fake_engine_unused(void) { fake_engine_unresolved(); }
