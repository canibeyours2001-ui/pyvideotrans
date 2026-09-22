import Studio from '@/components/studio/studio';
import {requireChatGPTUser} from './chatgpt-auth';
export const dynamic='force-dynamic';
export default async function Home(){await requireChatGPTUser('/');return <Studio/>}
