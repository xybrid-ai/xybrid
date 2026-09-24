// Reads the TypeScript side of the contract with the compiler API: the
// methods (and parameter names) of the Codegen `Spec`, and the shape of the
// public types exported from src/index.ts.

import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const root = fileURLToPath(new URL('../../', import.meta.url));

export function loadTsSurface() {
  const program = ts.createProgram([`${root}src/index.ts`], {
    strict: true,
    skipLibCheck: true,
    moduleResolution: ts.ModuleResolutionKind.Node10,
    target: ts.ScriptTarget.ES2020,
  });
  const checker = program.getTypeChecker();

  const specFile = program.getSourceFile(`${root}src/NativeXybrid.ts`);
  const spec = specFile.statements.find(
    (statement) => ts.isInterfaceDeclaration(statement) && statement.name.text === 'Spec',
  );
  const specMethods = new Map(
    spec.members
      .filter(ts.isMethodSignature)
      .map((method) => [method.name.getText(specFile), method.parameters.map((p) => p.name.getText(specFile))]),
  );

  const indexFile = program.getSourceFile(`${root}src/index.ts`);
  const exports = new Map(
    checker.getExportsOfModule(checker.getSymbolAtLocation(indexFile)).map((symbol) => [symbol.name, symbol]),
  );

  const resolve = (name) => {
    let symbol = exports.get(name);
    if (!symbol) return undefined;
    if (symbol.flags & ts.SymbolFlags.Alias) symbol = checker.getAliasedSymbol(symbol);
    return checker.getDeclaredTypeOfSymbol(symbol);
  };

  return {
    specMethods,
    /** Property names every member of the exported type has. */
    propertiesOf(name) {
      const type = resolve(name);
      if (!type) return undefined;
      return new Set(checker.getPropertiesOfType(type).map((property) => property.name));
    },
    /** String-literal members of an exported union type. */
    literalsOf(name) {
      const type = resolve(name);
      if (!type) return undefined;
      const members = type.isUnion() ? type.types : [type];
      return new Set(members.filter((member) => member.isStringLiteral()).map((member) => member.value));
    },
  };
}
