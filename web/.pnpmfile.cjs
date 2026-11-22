// openapi-typescript drives the TypeScript 5 compiler API, which the native TypeScript 7
// toolchain no longer ships. Give the generator its own TypeScript 5 instead of the
// project-level peer so `tsc --noEmit` keeps running on TypeScript 7.
module.exports = {
  hooks: {
    readPackage(pkg) {
      if (pkg.name === "openapi-typescript") {
        delete pkg.peerDependencies?.typescript;
        pkg.dependencies = { ...pkg.dependencies, typescript: "5.9.3" };
      }
      return pkg;
    },
  },
};
