/**
 * Offline syntax checker for the frontend sources.
 *
 * The sandbox has no network access, so `npm install` (and therefore `vite
 * build`) cannot run. This script performs the static checks that would
 * otherwise be caught by the bundler:
 *
 *   1. balanced braces / parentheses / brackets (outside strings, template
 *      literals and comments);
 *   2. balanced JSX tags - a real tag scanner that understands attributes
 *      containing `=>` and self-closing elements;
 *   3. every relative import resolves to a file that exists on disk;
 *   4. the entry point referenced by `index.html` exists.
 *
 * Usage: `node tests_local/check_frontend.js`
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const SRC = path.join(ROOT, 'frontend');

const OPENERS = { ')': '(', ']': '[', '}': '{' };

/**
 * Remove strings, template literals and comments.
 *
 * An apostrophe that directly follows a word character is treated as an
 * apostrophe (JSX text such as "d'objets"), not as the start of a string.
 */
function stripLiterals(source) {
  let out = '';
  let i = 0;
  const n = source.length;

  while (i < n) {
    const ch = source[i];
    const next = source[i + 1];

    if (ch === '/' && next === '/') {
      while (i < n && source[i] !== '\n') i += 1;
      continue;
    }
    if (ch === '/' && next === '*') {
      i += 2;
      while (i < n && !(source[i] === '*' && source[i + 1] === '/')) i += 1;
      i += 2;
      continue;
    }

    if (ch === '"' || ch === '`') {
      const quote = ch;
      i += 1;
      while (i < n) {
        if (source[i] === '\\') {
          i += 2;
          continue;
        }
        if (source[i] === quote) {
          i += 1;
          break;
        }
        i += 1;
      }
      out += '""';
      continue;
    }

    if (ch === "'") {
      const previous = i > 0 ? source[i - 1] : '';
      if (/[A-Za-z0-9]/.test(previous)) {
        // Apostrophe inside a word (JSX text): not a string delimiter.
        out += ' ';
        i += 1;
        continue;
      }
      i += 1;
      while (i < n) {
        if (source[i] === '\\') {
          i += 2;
          continue;
        }
        if (source[i] === "'") {
          i += 1;
          break;
        }
        i += 1;
      }
      out += '""';
      continue;
    }

    out += ch;
    i += 1;
  }
  return out;
}

function checkBalanced(code, file, failures) {
  const stack = [];
  for (const ch of code) {
    if (ch === '(' || ch === '[' || ch === '{') stack.push(ch);
    else if (ch === ')' || ch === ']' || ch === '}') {
      const open = stack.pop();
      if (open !== OPENERS[ch]) {
        failures.push(`${file}: unbalanced '${ch}' (expected '${OPENERS[ch]}')`);
        return;
      }
    }
  }
  if (stack.length) {
    failures.push(`${file}: ${stack.length} unclosed '${stack[stack.length - 1]}'`);
  }
}

/**
 * Scan the JSX elements of a file.
 *
 * The scanner understands:
 *   - `<Tag attr="x" />`     self closing (nothing is pushed);
 *   - `<Tag attr="x">`       pushes the element;
 *   - `</Tag>`               pops the element (the name must match);
 *   - `<>` / `</>`           fragments (ignored, they never nest);
 *   - attributes containing `=>` or `{}` (the `>` of an arrow function is not
 *     the end of the tag).
 */
function checkJsxTags(code, file, failures) {
  const stack = [];
  let i = 0;
  const n = code.length;
  let sawJsx = false;

  const skipToTagEnd = (start) => {
    // Returns the index of the character that closes the tag, plus whether
    // the tag was self-closing.
    let depth = 0;
    let j = start;
    while (j < n) {
      const c = code[j];
      if (c === '{' || c === '[' || c === '(') depth += 1;
      else if (c === '}' || c === ']' || c === ')') depth -= 1;
      else if (depth === 0 && c === '/' && code[j + 1] === '>') return { end: j, selfClosing: true };
      else if (depth === 0 && c === '>') return { end: j, selfClosing: false };
      else if (depth === 0 && c === '<' && code[j + 1] === '/') return { end: j, selfClosing: false };
      j += 1;
    }
    return { end: n, selfClosing: false };
  };

  while (i < n) {
    if (code[i] !== '<') {
      i += 1;
      continue;
    }

    // Fragment
    if (code[i + 1] === '>') {
      sawJsx = true;
      i += 2;
      continue;
    }
    if (code[i + 1] === '/') {
      // Closing tag
      if (code[i + 2] === '>') {
        i += 3; // fragment end
        continue;
      }
      const match = /^<\/\s*([A-Za-z][A-Za-z0-9_.]*)/.exec(code.slice(i));
      if (!match) {
        i += 1;
        continue;
      }
      sawJsx = true;
      const name = match[1];
      const open = stack.pop();
      if (open !== name) {
        failures.push(
          `${file}: </${name}> closes <${open ?? 'nothing'}> (line ${code.slice(0, i).split('\n').length})`,
        );
        return;
      }
      const { end } = skipToTagEnd(i + match[0].length);
      i = end + 1;
      continue;
    }

    // Opening tag
    const match = /^<\s*([A-Za-z][A-Za-z0-9_.]*)/.exec(code.slice(i));
    if (!match) {
      i += 1;
      continue;
    }
    sawJsx = true;
    const name = match[1];
    const { end, selfClosing } = skipToTagEnd(i + match[0].length);
    if (!selfClosing) stack.push(name);
    i = end + 1;
  }

  if (stack.length) {
    failures.push(`${file}: unclosed JSX element <${stack[stack.length - 1]}>`);
  }
  if (!sawJsx && file.endsWith('.jsx')) {
    failures.push(`${file}: no JSX element found - the file looks empty`);
  }
}

function checkImports(code, file, dir, failures) {
  const importPattern = /from\s+['"]([^'"]+)['"]/g;
  let match;
  while ((match = importPattern.exec(code)) !== null) {
    const specifier = match[1];
    if (!specifier.startsWith('.')) continue;
    const target = path.resolve(dir, specifier);
    if (!fs.existsSync(target)) {
      failures.push(`${file}: import '${specifier}' does not exist`);
    }
  }
}

function collectFiles(dir, extensions, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === 'node_modules' || entry.name === 'dist') continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) collectFiles(full, extensions, acc);
    else if (extensions.some((ext) => entry.name.endsWith(ext))) acc.push(full);
  }
  return acc;
}

function main() {
  const failures = [];
  const files = collectFiles(SRC, ['.js', '.jsx', '.html', '.css']);

  for (const file of files) {
    const relative = path.relative(ROOT, file);
    const source = fs.readFileSync(file, 'utf8');
    const code = stripLiterals(source);

    checkBalanced(code, relative, failures);
    if (file.endsWith('.jsx')) checkJsxTags(code, relative, failures);
    if (file.endsWith('.js') || file.endsWith('.jsx')) {
      checkImports(code, relative, path.dirname(file), failures);
    }
  }

  const html = fs.readFileSync(path.join(SRC, 'index.html'), 'utf8');
  const entryMatch = html.match(/src="([^"]+)"/);
  if (!entryMatch) {
    failures.push('frontend/index.html: no <script src="..."> entry point');
  } else {
    const target = path.join(SRC, entryMatch[1].replace(/^\//, ''));
    if (!fs.existsSync(target)) {
      failures.push(`frontend/index.html: entry point '${entryMatch[1]}' does not exist`);
    }
  }

  console.log(`Checked ${files.length} frontend files.`);
  if (failures.length) {
    console.log(`\nFAILED (${failures.length}):`);
    failures.forEach((item) => console.log(`  - ${item}`));
    process.exit(1);
  }
  console.log('ALL FRONTEND CHECKS PASSED');
}

main();
