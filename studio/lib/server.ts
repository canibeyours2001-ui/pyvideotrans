import {env} from 'cloudflare:workers';
export type Config={url:string;key:string};
const runtime=env as unknown as {DB:D1Database;SETTINGS_ENCRYPTION_KEY?:string};
export function identity(request:Request){
 const owner=request.headers.get('oai-authenticated-user-id');
 if(!owner)throw new Error('AUTH');
 if(!['GET','HEAD'].includes(request.method)){
  const origin=request.headers.get('origin');
  if(!origin||origin!==new URL(request.url).origin)throw new Error('ORIGIN');
 }
 return owner;
}
export async function ownerHash(owner:string){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(owner)))).map(x=>x.toString(16).padStart(2,'0')).join('')}
async function cryptoKey(){
 const secret=runtime.SETTINGS_ENCRYPTION_KEY;
 if(!secret||secret.length<32)throw new Error('SETUP');
 return crypto.subtle.importKey('raw',await crypto.subtle.digest('SHA-256',new TextEncoder().encode(secret)),{name:'AES-GCM'},false,['encrypt','decrypt']);
}
export async function config(owner:string):Promise<Config|null>{
 const row=await runtime.DB.prepare('SELECT url, encrypted_key FROM connections WHERE owner = ?').bind(owner).first<{url:string;encrypted_key:string}>();
 if(!row)return null;
 const bytes=Uint8Array.from(atob(row.encrypted_key),c=>c.charCodeAt(0));
 const plain=await crypto.subtle.decrypt({name:'AES-GCM',iv:bytes.slice(0,12)},await cryptoKey(),bytes.slice(12));
 return {url:row.url,key:new TextDecoder().decode(plain)};
}
export function validBackend(value:string){
 const url=new URL(value);
 if(url.protocol!=='https:'||url.username||url.password||url.port||url.search||url.hash||!url.hostname.endsWith('.modal.run')||url.pathname!=='/')throw new Error('Use the HTTPS Modal endpoint URL ending in .modal.run.');
 return url.origin;
}
export async function saveConfig(owner:string,url:string,key:string){
 const iv=crypto.getRandomValues(new Uint8Array(12));
 const encrypted=new Uint8Array(await crypto.subtle.encrypt({name:'AES-GCM',iv},await cryptoKey(),new TextEncoder().encode(key)));
 const bytes=new Uint8Array(iv.length+encrypted.length);bytes.set(iv);bytes.set(encrypted,iv.length);
 await runtime.DB.prepare('INSERT INTO connections (owner,url,encrypted_key) VALUES (?,?,?) ON CONFLICT(owner) DO UPDATE SET url=excluded.url,encrypted_key=excluded.encrypted_key').bind(owner,url,btoa(String.fromCharCode(...bytes))).run();
}
export function failure(e:unknown){
 const code=(e as Error).message;
 return Response.json({error:code==='AUTH'?'Sign in to continue.':code==='ORIGIN'?'This request must come from your studio.':code==='SETUP'?'Server encryption is not configured yet.':'The service is unavailable. Please retry.'},{status:code==='AUTH'?401:code==='ORIGIN'?403:503,headers:{'Cache-Control':'no-store'}});
}
export async function backendFetch(c:Config,owner:string,path:string,init:RequestInit={}){
 const headers=new Headers(init.headers);headers.set('Authorization','Bearer '+c.key);headers.set('X-Studio-Owner',await ownerHash(owner));
 return fetch(c.url+path,{...init,headers,redirect:'error',signal:AbortSignal.timeout(path==='/jobs'&&init.method==='POST'?330000:60000)});
}
