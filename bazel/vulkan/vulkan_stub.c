// Link-time stub for the Vulkan loader. NEVER shipped, never called.
//
// ggml's Vulkan backend routes almost everything through vulkan-hpp's dynamic
// dispatcher, so the static archives leave exactly three loader symbols
// undefined (grep ggml-vulkan.cpp for `vk[A-Z]`):
//
//   vkGetInstanceProcAddr          bootstraps the dispatcher
//   vkGetPhysicalDeviceFeatures2   queried before the dispatcher covers it
//   vkCmdCopyBuffer                one direct command call
//
// The obvious way to satisfy them is to link the exec machine's libvulkan.so,
// and that does not work here: it is built against ITS glibc, so resolving it
// against hermetic-llvm's 2.28 sysroot leaves the loader's own imports
// undefined (`__isoc23_sscanf@GLIBC_2.38`), and a host path is not an action
// input, so it is simply absent when the link runs on a remote worker.
//
// A stub compiled by our own toolchain fixes both. It carries the real
// library's SONAME (libvulkan.so.1, set in BUILD.bazel), so the produced binary
// records that in DT_NEEDED and the dynamic loader binds these calls to the
// machine's REAL Vulkan loader at startup. The stub bodies below exist only to
// give the linker something to resolve against; nothing ever executes them.
//
// Adding a fourth direct call upstream fails the link with the missing symbol
// named, which is the diagnostic we want — so this list stays honest rather
// than padded with entry points we do not use.

void *vkGetInstanceProcAddr(void *instance, const char *name) {
  (void)instance;
  (void)name;
  return 0;
}

void vkGetPhysicalDeviceFeatures2(void *physical_device, void *features) {
  (void)physical_device;
  (void)features;
}

void vkCmdCopyBuffer(void *command_buffer, void *src, void *dst,
                     unsigned int region_count, const void *regions) {
  (void)command_buffer;
  (void)src;
  (void)dst;
  (void)region_count;
  (void)regions;
}
