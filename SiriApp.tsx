import { useCallback, useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { AudioLines, Bot, Check, ExternalLink, Loader2, Mic, MicOff, Send, Settings2 } from 'lucide-react';
import { useMacOS } from '@/contexts/MacOSContext';

type Message = { id: string; role: 'user' | 'assistant'; content: string; links?: { title: string; url: string; snippet: string }[]; searchQuery?: string };
type SpeechRecognitionAlternative = { transcript: string };
type SpeechRecognitionResult = { 0: SpeechRecognitionAlternative; isFinal: boolean };
type SpeechRecognitionEvent = { results: ArrayLike<SpeechRecognitionResult> };
type SpeechRecognitionLike = {
  continuous: boolean; interimResults: boolean; lang: string;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void; stop: () => void;
};
type WindowWithSpeech = Window & { SpeechRecognition?: new () => SpeechRecognitionLike; webkitSpeechRecognition?: new () => SpeechRecognitionLike };
type Generator = (prompt: string, options: Record<string, unknown>) => Promise<Array<{ generated_text: string }>>;

const MODEL_ID = 'onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX';
const HISTORY_KEY = 'thanasos-siri-history-v1';
const SYSTEM_PROMPT = 'You are Siri, the helpful assistant built into ThanasOS. Be friendly, concise, accurate, and practical. You can help operate this simulated macOS desktop. Do not claim to have performed actions unless the system confirms them. You are a small local model; if unsure, say so.';

let generatorPromise: Promise<Generator> | null = null;
async function getGenerator(onProgress: (value: number) => void): Promise<Generator> {
  if (!generatorPromise) {
    generatorPromise = (async () => {
      const moduleUrl = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';
      // Loaded from the official Transformers.js package CDN so the OS can remain a simple static website.
      const transformers = await import(/* @vite-ignore */ moduleUrl) as {
        env: { useBrowserCache: boolean; allowLocalModels: boolean };
        pipeline: (task: string, model: string, options: Record<string, unknown>) => Promise<Generator>;
      };
      transformers.env.useBrowserCache = true;
      transformers.env.allowLocalModels = false;
      const progress_callback = (event: { progress?: number; status?: string }) => {
        if (typeof event.progress === 'number') onProgress(Math.max(0, Math.min(100, event.progress)));
      };
      try {
        return await transformers.pipeline('text-generation', MODEL_ID, { dtype: 'q4', device: 'webgpu', progress_callback });
      } catch {
        return await transformers.pipeline('text-generation', MODEL_ID, { dtype: 'q4', device: 'wasm', progress_callback });
      }
    })().catch(error => { generatorPromise = null; throw error; });
  }
  return generatorPromise;
}

const makeMessage = (role: Message['role'], content: string, links?: Message['links'], searchQuery?: string): Message => ({ id: `${Date.now()}-${Math.random()}`, role, content, links, searchQuery });

export const SiriApp = () => {
  const { apps, openApp, openUrl, settings, updateSettings, minimizeAllWindows } = useMacOS();
  const [messages, setMessages] = useState<Message[]>(() => {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]').slice(-30); } catch { return []; }
  });
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelProgress, setModelProgress] = useState(0);
  const [modelError, setModelError] = useState('');
  const [listening, setListening] = useState(false);
  const [voiceReplies, setVoiceReplies] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [wakeMode, setWakeMode] = useState(false);
  const [online, setOnline] = useState(navigator.onLine);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);

  useEffect(() => { localStorage.setItem(HISTORY_KEY, JSON.stringify(messages.slice(-30))); }, [messages]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, busy]);
  useEffect(() => {
    const onOnline = () => setOnline(true); const onOffline = () => setOnline(false);
    window.addEventListener('online', onOnline); window.addEventListener('offline', onOffline);
    return () => { window.removeEventListener('online', onOnline); window.removeEventListener('offline', onOffline); };
  }, []);
  useEffect(() => () => { recognitionRef.current?.stop(); window.speechSynthesis?.cancel(); }, []);

  const say = useCallback((text: string) => {
    if (!voiceReplies || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1; utterance.pitch = 1.04;
    window.speechSynthesis.speak(utterance);
  }, [voiceReplies]);

  const loadModel = async () => {
    if (modelReady || modelLoading) return;
    setModelLoading(true); setModelError('');
    try {
      await getGenerator(setModelProgress);
      setModelReady(true); setModelProgress(100);
      setMessages(prev => [...prev, makeMessage('assistant', 'I’m ready! My 1.1-billion-parameter model is loaded on this device. After the first download, the browser can reuse its cached model files.')]);
    } catch (error) {
      setModelError(error instanceof Error ? error.message : 'The model could not be loaded. Check your connection and available memory, then try again.');
    } finally { setModelLoading(false); }
  };

  const searchWeb = async (query: string) => {
    const params = new URLSearchParams({ action: 'query', list: 'search', srsearch: query, format: 'json', origin: '*', srlimit: '5' });
    const response = await fetch(`https://en.wikipedia.org/w/api.php?${params.toString()}`);
    if (!response.ok) throw new Error('The web search service did not respond.');
    const data = await response.json();
    const links = (data.query?.search || []).map((item: { title: string; snippet: string; pageid: number }) => ({
      title: item.title,
      url: `https://en.wikipedia.org/?curid=${item.pageid}`,
      snippet: String(item.snippet || '').replace(/<[^>]*>/g, '').replace(/&quot;/g, '"').replace(/&#039;/g, "'").replace(/&amp;/g, '&'),
    }));
    return links as { title: string; url: string; snippet: string }[];
  };

  const performCommand = async (raw: string): Promise<string | null> => {
    const text = raw.trim();
    const lower = text.toLowerCase();
    const app = apps.find(item => lower === `open ${item.name.toLowerCase()}` || lower === `launch ${item.name.toLowerCase()}` || lower === `start ${item.name.toLowerCase()}`);
    if (app) { openApp(app.id); return `Opening ${app.name}.`; }
    const openMatch = lower.match(/^(?:please\s+)?(?:can you\s+)?(?:open|launch|start)\s+(.+?)[?.!]*$/);
    if (openMatch) {
      const match = apps.find(item => item.name.toLowerCase().includes(openMatch[1]) || item.id.toLowerCase() === openMatch[1]);
      if (match) { openApp(match.id); return `Opening ${match.name}.`; }
    }
    if (/^(?:show desktop|minimize all windows|hide all windows)$/.test(lower)) { minimizeAllWindows(); return 'I’ve shown the desktop.'; }
    if (/^(?:turn|make) (?:the )?volume up$/.test(lower)) { const volume = Math.min(100, (settings.volume ?? 65) + 10); updateSettings({ volume }); return `Volume increased to ${volume} percent.`; }
    if (/^(?:turn|make) (?:the )?volume down$/.test(lower)) { const volume = Math.max(0, (settings.volume ?? 65) - 10); updateSettings({ volume }); return `Volume decreased to ${volume} percent.`; }
    const volumeMatch = lower.match(/(?:set|change) (?:the )?volume to (\d{1,3})/);
    if (volumeMatch) { const volume = Math.max(0, Math.min(100, Number(volumeMatch[1]))); updateSettings({ volume }); return `Volume set to ${volume} percent.`; }
    if (/^(?:turn|switch) (?:the )?wi-?fi on$/.test(lower)) { updateSettings({ wifi: true }); return 'Wi-Fi is switched on in ThanasOS.'; }
    if (/^(?:turn|switch) (?:the )?wi-?fi off$/.test(lower)) { updateSettings({ wifi: false }); return 'Wi-Fi is switched off in ThanasOS.'; }
    if (/^(?:turn|switch) (?:the )?bluetooth on$/.test(lower)) { updateSettings({ bluetooth: true }); return 'Bluetooth is switched on in ThanasOS.'; }
    if (/^(?:turn|switch) (?:the )?bluetooth off$/.test(lower)) { updateSettings({ bluetooth: false }); return 'Bluetooth is switched off in ThanasOS.'; }
    if (/^(?:turn on|enable) dark mode$/.test(lower)) { updateSettings({ theme: 'dark' }); return 'Dark appearance is on.'; }
    if (/^(?:turn on|enable) light mode$/.test(lower)) { updateSettings({ theme: 'light' }); return 'Light appearance is on.'; }
    if (/^(?:what time is it|what's the time|tell me the time)\??$/.test(lower)) return `It is ${new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date())}.`;
    if (/^(?:what day is it|what's the date|what is the date)\??$/.test(lower)) return `Today is ${new Intl.DateTimeFormat(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' }).format(new Date())}.`;
    const searchMatch = lower.match(/^(?:search(?: the web)? for|google|look up|find on the web)\s+(.+)$/);
    if (searchMatch) {
      const query = text.replace(/^(?:search(?: the web)? for|google|look up|find on the web)\s+/i, '').trim();
      try {
        const results = await searchWeb(query);
        const summary = results.length ? `I found ${results.length} Wikipedia search results for “${query}”. You can open one below, or search the wider web with Google.` : `I couldn't find matching Wikipedia pages for “${query}”. You can still search the wider web with Google.`;
        return JSON.stringify({ __search: true, summary, query, results });
      } catch {
        return JSON.stringify({ __search: true, summary: `I couldn't reach the search service just now. You can still search Google for “${query}”.`, query, results: [] });
      }
    }
    const webMatch = lower.match(/^(?:open website|go to website|visit)\s+(.+)$/);
    if (webMatch) {
      let url = text.slice(text.toLowerCase().indexOf(webMatch[1])).trim();
      if (!/^https?:\/\//i.test(url)) url = `https://${url}`;
      openUrl(url); return `Opening ${url} in Safari.`;
    }
    return null;
  };

  const send = async (value = input) => {
    const prompt = value.trim(); if (!prompt || busy) return;
    setInput(''); setBusy(true);
    setMessages(prev => [...prev, makeMessage('user', prompt)]);
    try {
      const command = await performCommand(prompt);
      if (command?.startsWith('{"__search"')) {
        const parsed = JSON.parse(command) as { summary: string; query: string; results: Message['links'] };
        const reply = makeMessage('assistant', parsed.summary, parsed.results, parsed.query);
        setMessages(prev => [...prev, reply]);
        return;
      }
      if (command) { setMessages(prev => [...prev, makeMessage('assistant', command)]); say(command); return; }
      if (!modelReady) {
        const reply = online
          ? 'I can control ThanasOS and search the web right now. To answer general questions privately on this device, load my local 1.1B model using the button above.'
          : 'I’m offline. I can still control the apps and settings in ThanasOS, but load my local model before disconnecting if you want offline chat.';
        setMessages(prev => [...prev, makeMessage('assistant', reply)]); say(reply); return;
      }
      const generator = await getGenerator(setModelProgress);
      const history = [...messages.slice(-6), makeMessage('user', prompt)].map(message => `${message.role === 'user' ? '<|user|>' : '<|assistant|>'}\n${message.content}`).join('\n');
      const generated = await generator(`<|system|>\n${SYSTEM_PROMPT}</s>\n${history}\n<|assistant|>\n`, { max_new_tokens: 180, temperature: 0.7, do_sample: true, top_p: 0.9, repetition_penalty: 1.12, return_full_text: false });
      const answer = String(generated?.[0]?.generated_text || '').split('</s>')[0].trim() || 'I’m sorry, I couldn’t come up with an answer. Could you try asking another way?';
      setMessages(prev => [...prev, makeMessage('assistant', answer)]); say(answer);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Something went wrong.';
      setMessages(prev => [...prev, makeMessage('assistant', `Sorry — I hit a problem: ${message}`)]);
    } finally { setBusy(false); }
  };

  const startListening = (wake = false) => {
    const SpeechRecognition = (window as WindowWithSpeech).SpeechRecognition || (window as WindowWithSpeech).webkitSpeechRecognition;
    if (!SpeechRecognition) { setMessages(prev => [...prev, makeMessage('assistant', 'Voice input is not supported by this browser. Try Chrome or type your request instead.')]); return; }
    recognitionRef.current?.stop();
    const recognition = new SpeechRecognition(); recognition.lang = 'en-AU'; recognition.continuous = wake; recognition.interimResults = false;
    recognition.onresult = event => {
      const last = event.results[event.results.length - 1];
      if (!last?.isFinal) return;
      let transcript = last[0].transcript.trim();
      if (wake) {
        const match = transcript.match(/hey siri[,. ]*(.*)/i);
        if (!match) return;
        transcript = match[1].trim() || 'Hello Siri';
        setWakeMode(false); recognition.stop(); setListening(false); void send(transcript); return;
      }
      setListening(false); void send(transcript);
    };
    recognition.onerror = () => { setListening(false); setWakeMode(false); };
    recognition.onend = () => { setListening(false); if (wake) setWakeMode(false); };
    recognitionRef.current = recognition;
    try { recognition.start(); setListening(true); setWakeMode(wake); } catch { setListening(false); }
  };

  const clearHistory = () => { setMessages([]); localStorage.removeItem(HISTORY_KEY); window.speechSynthesis?.cancel(); };

  return (
    <div className="h-full w-full flex flex-col text-white overflow-hidden" style={{ background: 'radial-gradient(ellipse at 50% -15%, rgba(89,91,190,.34), transparent 55%), linear-gradient(145deg,#171722,#090a12 70%)' }}>
      <header className="flex items-center justify-between px-5 py-4 border-b border-white/10 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg,#6d82ff,#c46cff)', boxShadow: '0 5px 24px #7c68ff55' }}><AudioLines className="w-6 h-6" /></div>
          <div><div className="font-semibold tracking-wide">Siri</div><div className="text-[11px] text-white/50">ThanasOS assistant</div></div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`hidden sm:flex items-center gap-1.5 text-[11px] ${online ? 'text-emerald-300' : 'text-amber-300'}`}><span className={`w-1.5 h-1.5 rounded-full ${online ? 'bg-emerald-400' : 'bg-amber-400'}`} />{online ? 'Online' : 'Offline'}</span>
          <button onClick={() => setShowSettings(v => !v)} title="Siri settings" className="p-2 rounded-lg hover:bg-white/10"><Settings2 className="w-4 h-4" /></button>
        </div>
      </header>

      {showSettings && <div className="px-5 py-3 border-b border-white/10 bg-white/[.04] flex flex-wrap items-center gap-4 text-xs">
        <label className="flex items-center gap-2 cursor-pointer"><input type="checkbox" checked={voiceReplies} onChange={e => setVoiceReplies(e.target.checked)} /> Speak replies</label>
        <button onClick={clearHistory} className="text-white/65 hover:text-white">Clear conversation</button>
        <span className="text-white/40">Model files are cached by your browser. Voice dictation uses your browser’s speech service.</span>
      </div>}

      <main className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-7 py-5">
        {messages.length === 0 && <div className="max-w-xl mx-auto text-center pt-5 sm:pt-12">
          <motion.div animate={{ scale: [1, 1.04, 1], opacity: [0.85, 1, 0.85] }} transition={{ duration: 3.2, repeat: Infinity }} className="mx-auto mb-6 w-24 h-24 rounded-full flex items-center justify-center" style={{ background: 'radial-gradient(circle at 30% 25%,#a7b6ff,#7462e8 48%,#302b68)', boxShadow: '0 0 70px #8170ff66, inset 0 2px 14px #ffffff77' }}><AudioLines className="w-12 h-12" /></motion.div>
          <h1 className="text-2xl sm:text-3xl font-semibold mb-2">What can I help you with?</h1>
          <p className="text-sm text-white/50 mb-7">Ask me to open apps, change settings, search the web, or chat privately with my on-device model.</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-left">
            {['Open Safari','Turn dark mode on','Search the web for space news','What time is it?'].map(suggestion => <button key={suggestion} onClick={() => void send(suggestion)} className="rounded-xl border border-white/10 bg-white/[.045] hover:bg-white/10 px-4 py-3 text-sm text-white/80 transition-colors">{suggestion}</button>)}
          </div>
        </div>}
        <div className="max-w-3xl mx-auto space-y-5">
          {messages.map(message => <div key={message.id} className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {message.role === 'assistant' && <div className="w-8 h-8 rounded-xl shrink-0 flex items-center justify-center bg-gradient-to-br from-indigo-400 to-violet-600"><AudioLines className="w-4 h-4" /></div>}
            <div className={`max-w-[88%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${message.role === 'user' ? 'bg-indigo-500/80 text-white rounded-br-md' : 'bg-white/[.075] border border-white/[.07] text-white/90 rounded-bl-md'}`}>
              <p className="whitespace-pre-wrap">{message.content}</p>
              {((message.links && message.links.length > 0) || message.searchQuery) && <div className="mt-3 space-y-2">
                {(message.links || []).map(link => <a key={link.url} href={link.url} target="_blank" rel="noreferrer" className="block rounded-lg bg-black/20 hover:bg-black/35 p-3 transition-colors"><span className="font-medium text-indigo-200 flex items-center gap-1">{link.title}<ExternalLink className="w-3 h-3" /></span><span className="block mt-1 text-xs text-white/55">{link.snippet}</span></a>)}
                <button onClick={() => openUrl(`https://www.google.com/search?q=${encodeURIComponent(message.searchQuery || message.content)}`)} className="text-xs text-indigo-200 hover:text-white flex items-center gap-1">Search the wider web with Google <ExternalLink className="w-3 h-3" /></button>
              </div>}
            </div>
          </div>)}
          {busy && <div className="flex items-center gap-3 text-sm text-white/55"><div className="w-8 h-8 rounded-xl flex items-center justify-center bg-white/10"><Loader2 className="w-4 h-4 animate-spin" /></div>Siri is thinking…</div>}
          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="shrink-0 border-t border-white/10 px-4 sm:px-6 pt-3 pb-4 bg-black/15">
        <div className="max-w-3xl mx-auto">
          {!modelReady && <div className="mb-3 flex flex-wrap items-center gap-3 rounded-xl border border-indigo-300/15 bg-indigo-400/[.07] px-3 py-2.5">
            <div className="flex-1 min-w-[170px]"><div className="text-xs font-medium flex items-center gap-1.5"><Bot className="w-3.5 h-3.5 text-indigo-200" />Local AI · TinyLlama 1.1B</div><div className="text-[10px] text-white/45 mt-0.5">Optional first download, roughly 910 MB. Runs on this device.</div>{modelLoading && <div className="h-1 rounded-full bg-white/10 mt-2 overflow-hidden"><div className="h-full bg-indigo-400 transition-all" style={{ width: `${modelProgress}%` }} /></div>}{modelError && <p className="text-[10px] text-rose-300 mt-1">{modelError}</p>}</div>
            <button onClick={() => void loadModel()} disabled={modelLoading} className="rounded-lg px-3 py-2 text-xs font-medium bg-indigo-500 hover:bg-indigo-400 disabled:opacity-50 flex items-center gap-1.5">{modelLoading ? <><Loader2 className="w-3.5 h-3.5 animate-spin" />{Math.round(modelProgress)}%</> : <><DownloadIcon />Load local model</>}</button>
          </div>}
          {modelReady && <div className="mb-2 text-[10px] text-emerald-300/75 flex items-center gap-1"><Check className="w-3 h-3" />1.1B local model ready · cached in this browser</div>}
          <form onSubmit={e => { e.preventDefault(); void send(); }} className="flex items-center gap-2 rounded-2xl border border-white/15 bg-white/[.07] p-2 focus-within:border-indigo-300/50 transition-colors">
            <button type="button" title={listening ? 'Stop listening' : 'Speak to Siri'} onClick={() => listening ? recognitionRef.current?.stop() : startListening(false)} className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${listening ? 'bg-rose-500 text-white' : 'hover:bg-white/10 text-white/75'}`}>{listening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}</button>
            <input ref={inputRef} value={input} onChange={e => setInput(e.target.value)} placeholder={listening ? 'Listening…' : 'Ask Siri anything…'} className="flex-1 min-w-0 bg-transparent outline-none text-sm text-white placeholder-white/35" disabled={busy} />
            <button type="submit" disabled={!input.trim() || busy} className="w-10 h-10 rounded-xl flex items-center justify-center bg-indigo-500 hover:bg-indigo-400 disabled:opacity-30 transition-colors"><Send className="w-4 h-4" /></button>
          </form>
          <div className="mt-2 flex items-center justify-between text-[10px] text-white/35"><span>{listening ? 'Listening for your voice' : 'Enter to send · Option + Space opens Siri'}</span><button onClick={() => wakeMode ? recognitionRef.current?.stop() : startListening(true)} className={`flex items-center gap-1 hover:text-white/70 ${wakeMode ? 'text-indigo-200' : ''}`}><Mic className="w-3 h-3" />{wakeMode ? 'Hey Siri listening' : 'Enable Hey Siri'}</button></div>
        </div>
      </footer>
    </div>
  );
};

const DownloadIcon = () => <span className="text-sm leading-none">↓</span>;
