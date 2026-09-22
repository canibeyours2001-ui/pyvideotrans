import {identity,config,backendFetch,failure} from '@/lib/server';
async function handle(request:Request){try{
 const owner=identity(request),c=await config(owner);
 if(!c)return request.method==='GET'&&new URL(request.url).pathname==='/api/jobs'?Response.json({jobs:[]}):Response.json({error:'Connect a processing backend in Connections first.'},{status:503});
 const path=new URL(request.url).pathname.replace(/^\/api/,'');
 if(!/^\/jobs(?:\/[a-f0-9]{32}(?:\/(?:render|cancel|retry|files\/(?:subtitles\.srt|subtitles\.vtt|segments\.json|output\.mp4)))?)?$/.test(path))return Response.json({error:'Not found'},{status:404});
 const headers=new Headers();for(const name of ['Content-Type','Content-Length','X-Video-Name','X-Video-Options','X-Request-Id']){const v=request.headers.get(name);if(v)headers.set(name,v)}
 if(request.method==='POST'&&path==='/jobs'&&Number(headers.get('Content-Length')||0)>=100000000)return Response.json({error:'Video must be smaller than 100 MB.'},{status:413});
 const response=await backendFetch(c,owner,path,{method:request.method,headers,body:['GET','HEAD'].includes(request.method)?undefined:request.body});
 const out=new Headers({'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});for(const name of ['Content-Type','Content-Disposition']){const v=response.headers.get(name);if(v)out.set(name,v)}
 return new Response(response.body,{status:response.status,headers:out});
 }catch(e){return failure(e)}}
export {handle as GET,handle as POST,handle as PATCH,handle as DELETE};
