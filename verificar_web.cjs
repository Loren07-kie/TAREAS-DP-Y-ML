const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes=new Map();let lastExamples=null,fillCalls=0;
const graphics=new Proxy({fillRect(){fillCalls++}},{get(target,key){return target[key]||(()=>{})}});
class Element{
  constructor(key=''){this.key=key;this.value='';this.dataset={};this.attrs={};this.children=[];this.listeners={};this.hidden=false;this._html='';this.textContent='';this.classList={contains(){return false}}}
  set innerHTML(value){this._html=value;this.items=[];for(const match of value.matchAll(/<(input|button)\b([^>]*)>/g)){const attrs=Object.fromEntries([...match[2].matchAll(/([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));const el=attrs.id?get('#'+attrs.id):new Element();el.tag=match[1];el.value=attrs.value||el.value;for(const [k,v] of Object.entries(attrs))if(k.startsWith('data-'))el.dataset[k.slice(5)]=v;this.items.push(el)}}
  get innerHTML(){return this._html}
  appendChild(el){this.children.push(el);lastExamples=el}
  querySelectorAll(selector){return (this.items||[]).filter(el=>selector==='input'?el.tag==='input':selector.includes('data-preset')?'preset'in el.dataset:selector.includes('data-answer')?'answer'in el.dataset:false)}
  addEventListener(key,fn){this.listeners[key]=fn}
  setAttribute(key,value){this.attrs[key]=value}
  getContext(){return graphics}
  getBoundingClientRect(){return {left:0,top:0,width:720,height:440}}
  scrollIntoView(){} focus(){}
}
function get(key){if(!nodes.has(key))nodes.set(key,new Element(key));return nodes.get(key)}
const document={querySelector:get,getElementById:id=>get('#'+id),createElement:()=>new Element(),querySelectorAll(selector){if(selector.includes('data-answer'))return get('#lesson').querySelectorAll(selector);if(selector==='#inputs input')return get('#inputs').querySelectorAll('input');return []}};
const context={document,console,Float32Array,Uint8Array,Math,Array,Number,Infinity,fetch(){throw Error('La página no debe descargar modelos al iniciar')}};
vm.createContext(context);vm.runInContext(fs.readFileSync('web/app.js','utf8'),context);
assert(get('#readout').innerHTML.includes('/ 20'));assert(!get('#readout').textContent.includes('Cargando'));
for(let n=1;n<=4;n++){
  vm.runInContext(`select(${n})`,context);
  assert(get('#lesson').innerHTML.includes('MINIRRETO'));
  assert.equal(lastExamples.querySelectorAll('[data-preset]').length,n===1||n===2?2:3);
  for(const example of lastExamples.querySelectorAll('[data-preset]'))example.onclick();
  const correct=vm.runInContext(`lessons[${n-1}].correct`,context);
  get('#lesson').querySelectorAll('[data-answer]')[correct].onclick();assert(get('#quiz-feedback').textContent.includes('Reto superado'));
}
vm.runInContext("select(3);let realInfer=infer;let calls=0;infer=(...args)=>{calls++;return realInfer(...args)};networkBackground=null;drawNetwork()",context);
const initial=vm.runInContext('calls',context);assert.equal(initial,6001);
for(let i=0;i<30;i++)vm.runInContext('drawNetwork()',context);
assert.equal(vm.runInContext('calls',context),initial+30);
assert.equal(get('#training').hidden,false);assert(get('#training').innerHTML.includes('600 points')===false);assert(get('#training').innerHTML.includes('600 puntos'));
get('#canvas').listeners.pointerdown({clientX:370,clientY:207.5});assert.equal(+get('#x1').value,0);assert.equal(+get('#x2').value,0);
const refs=JSON.parse(fs.readFileSync('web/modelos/referencia_inferencia.json','utf8'));
for(const row of refs)assert(Math.abs(vm.runInContext(`infer(${row.x},${row.y})`,context)-row.probabilidad)<1e-5);
assert.equal(vm.runInContext('components(.5).length',context),4);assert.equal(vm.runInContext('components(.8).length',context),2);assert.equal(vm.runInContext('components(1).length',context),0);
vm.runInContext("select(2);document.querySelector('#k').value=3;initK()",context);
let prev=Infinity;for(let i=0;i<12;i++){vm.runInContext('stepK()',context);const next=vm.runInContext('Z.reduce((s,p)=>s+p.reduce((a,v,j)=>a+(v-centers[closest(p)][j])**2,0),0)',context);assert(next<=prev+1e-8);prev=next}
const html=fs.readFileSync('web/index.html','utf8'),css=fs.readFileSync('web/style.css','utf8');
assert(html.includes('id="lesson"'));assert(!html.includes('✳'));assert(!css.includes('@import'));assert(!fs.readFileSync('web/app.js','utf8').includes('fetch('));
assert(html.includes('assets/orbita-logo.svg'));assert.equal((html.match(/role="tab"/g)||[]).length,4);
for(let i=1;i<=4;i++)assert(fs.existsSync(`web/descargas/Tarea_${i}_Semana_${String(i+8).padStart(2,'0')}.zip`));
console.log('OK: inicio sin descargas de modelos, 4 lecciones, 10 ejemplos, 4 retos, clic en gráfico, inferencia Keras/JS, caché del fondo, K-Means y visión.');
