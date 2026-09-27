import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";

export default tseslint.config(
  {
    ignores: [".next/**", "node_modules/**"],
  },
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{ts,tsx}"],
    plugins: {
      "react-hooks": reactHooks,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // Next.js App Router files routinely export non-component
      // values (metadata, config objects) alongside the default
      // component export — this is idiomatic, not a bug, so the
      // strict "unused vars must be prefixed with _" default is
      // relaxed for catch-block bindings we intentionally don't use.
      "@typescript-eslint/no-unused-vars": [
        "error",
        { caughtErrors: "none" },
      ],
    },
  }
);
