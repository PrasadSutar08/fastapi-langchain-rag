export type Token={access_token:string;token_type:string};
export type Conversation={id:number;title:string;created_at:string;updated_at:string};
export type Message={id:number;conversation_id:number;role:string;content:string;created_at:string};
export type DocumentItem={id:number;filename:string;mime_type:string;file_size:number;status:string;uploaded_at:string;last_indexed?:string;chunk_count:number};
export type Source={document_id?:number;filename?:string;page?:number;source?:string;chunk_index?:number};
export type SearchResult={document_id?:number;filename?:string;page?:number;content:string;score?:number};
const API_BASE=(import.meta.env.VITE_API_BASE_URL||"http://localhost:8000/api/v1").replace(/\/$/,"");
const TOKEN_KEY="rag_access_token";
export const getToken=()=>localStorage.getItem(TOKEN_KEY);
export const clearToken=()=>localStorage.removeItem(TOKEN_KEY);
async function request<T>(path:string,init:RequestInit={}):Promise<T>{const headers=new Headers(init.headers);const token=getToken();if(token)headers.set("Authorization",`Bearer ${token}`);if(!(init.body instanceof FormData)&&!headers.has("Content-Type"))headers.set("Content-Type","application/json");const response=await fetch(`${API_BASE}${path}`,{...init,headers});if(response.status===401){clearToken();window.dispatchEvent(new Event("auth-expired"));}if(!response.ok){let detail=`Request failed (${response.status})`;try{const body=await response.json();detail=body.detail||detail}catch{}throw new Error(detail)}if(response.status===204)return undefined as T;return response.json()}
export async function login(email:string,password:string){const body=new URLSearchParams({username:email,password});const result=await request<Token>("/login/access-token",{method:"POST",body,headers:{"Content-Type":"application/x-www-form-urlencoded"}});localStorage.setItem(TOKEN_KEY,result.access_token);return result}
export const listConversations=()=>request<{data:Conversation[]}>("/conversations");
export const createConversation=()=>request<{data:Conversation}>("/conversations",{method:"POST"});
export const renameConversation=(id:number,title:string)=>request<{data:Conversation}>(`/conversations/${id}`,{method:"PATCH",body:JSON.stringify({title})});
export const autoTitleConversation=(id:number,message:string)=>request<{data:Conversation}>(`/conversations/${id}/auto-title`,{method:"POST",body:JSON.stringify({message})});
export const deleteMessagesFrom=(conversationId:number,messageId:number)=>request<void>(`/conversations/${conversationId}/messages/from/${messageId}`,{method:"DELETE"});
export const deleteConversation=(id:number)=>request<void>(`/conversations/${id}`,{method:"DELETE"});
export const listMessages=(id:number)=>request<{data:Message[]}>(`/conversations/${id}/messages`);
export const chat=(conversation_id:number,message:string)=>request<{data:{conversation_id:number;answer:string;sources:Source[];user_message_id?:number;assistant_message_id?:number}}>("/qa/chat",{method:"POST",body:JSON.stringify({conversation_id,message})});
export const listDocuments=()=>request<{documents:DocumentItem[];total:number}>("/documents?skip=0&limit=100");
export const uploadDocument=(file:File)=>{const body=new FormData();body.append("file",file);return request("/documents/upload",{method:"POST",body})};
export const deleteDocument=(id:number)=>request<void>(`/documents/${id}`,{method:"DELETE"});
export const searchDocuments=(query:string)=>request<{query:string;results:SearchResult[]}>("/search",{method:"POST",body:JSON.stringify({query,top_k:5,search_type:"mmr"})});
