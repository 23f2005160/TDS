const fs = require('fs');
const sr = require(__dirname + "/seedrandom.js");
const ee = { default: (seed) => sr(seed, { global: false }) };

const we = "tds-ga4-q4-data-74b0cb0ad988a5d60aa486353b85d4ff816446657b041c85";
const ct = ["finance","engineering","marketing","sales","hr","legal"];
const lt = ["north_america","europe","asia_pacific","latin_america"];

function Dn(n){let o=String(n||"").trim().toLowerCase(),i=ee.default(`${we}#${o}#q-vector-search-rerank-api#data`),s=[],m={};for(let l=1;l<=500;l++){let e=`D${String(l).padStart(3,"0")}`,r=ct[Math.floor(i()*ct.length)],c=lt[Math.floor(i()*lt.length)],t=2020+Math.floor(i()*7);s.push({doc_id:e,title:`Document Title ${e} (${r})`,department:r,year:t,region:c,text:`This is the body text of document ${e} in department ${r} for region ${c} and year ${t}.`});let a=[],h=ee.default(`${we}#${o}#q4#doc#${e}`);for(let u=0;u<100;u++)a.push(parseFloat((h()*2-1).toFixed(4)));m[e]=a}let d={};for(let l=1;l<=10;l++){let e=`Q${String(l).padStart(3,"0")}`,r={},c=ee.default(`${we}#${o}#q4#query#${e}`);for(let t=1;t<=500;t++){let a=`D${String(t).padStart(3,"0")}`;r[a]=parseFloat(c().toFixed(4))}d[e]=r}return{documents:s,embeddings:m,reranker_scores:d}}

module.exports = { generateQ4: Dn };

if (require.main === module) {
  const email = (process.argv[2]||"").trim();
  if(!email){ console.error("Usage: node q4_generate.js <email>"); process.exit(1); }
  const m = Dn(email);
  // generate documents.csv
  let csv = [["doc_id","title","department","year","region","text"].join(","),
    ...m.documents.map(y=>[y.doc_id,`"${y.title}"`,y.department,y.year,y.region,`"${y.text}"`].join(","))].join("\n");
  fs.writeFileSync("documents.csv", csv);
  fs.writeFileSync("embeddings.json", JSON.stringify(m.embeddings,null,2));
  fs.writeFileSync("reranker_scores.json", JSON.stringify(m.reranker_scores,null,2));
  console.error("Q4 Data generated for", email);
}
