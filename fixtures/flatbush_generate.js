import Flatbush from "flatbush";
import { readFileSync, writeFileSync } from "fs";
import { dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
console.log(__dirname);

const buffer = readFileSync(`${__dirname}/data1_input.raw`);
const data = new Float64Array(
  buffer.buffer,
  buffer.byteOffset,
  buffer.byteLength / Float64Array.BYTES_PER_ELEMENT
);

const index = new Flatbush(data.length / 4);
for (let i = 0; i < data.length; i += 4) {
  index.add(data[i], data[i + 1], data[i + 2], data[i + 3]);
}

index.finish();

writeFileSync(`${__dirname}/data1_flatbush_js.raw`, new Uint8Array(index.data));
