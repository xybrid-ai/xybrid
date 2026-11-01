// Host JSON probe (JS): key order and number identity after JSON.parse.
// Usage: node probe.mjs "$(cat ../input.json)"
const o = JSON.parse(process.argv[2]);
console.log('order', Object.keys(o.questions));
console.log('state', JSON.stringify(o.state));
console.log('-inf', JSON.stringify(-Infinity));
const inserted = {};
for (const key of ['b', '2', 'a', '1']) inserted[key] = key;
console.log('insertion b,2,a,1 iterates as', Object.keys(inserted));
