import React from "react";
import { createRoot } from "react-dom/client";
import { MessageCirclePlus, PanelLeft, Search, Send, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL ?? "/api/chat";
const WELCOME_MESSAGE = "Hi, How can I help you today ? ";

function createChat(id) {
  return {
    id,
    title: `Chat ${id}`,
    messages: [{ role: "assistant", content: WELCOME_MESSAGE }]
  };
}

function MessageContent({ message }) {
  if (message.role === "assistant") {
    return <ReactMarkdown>{message.content}</ReactMarkdown>;
  }

  return message.content;
}

function App() {
  const [chatCounter, setChatCounter] = React.useState(1);
  const [activeChatId, setActiveChatId] = React.useState(1);
  const [chats, setChats] = React.useState([createChat(1)]);
  const [draft, setDraft] = React.useState("");
  const [isLoading, setIsLoading] = React.useState(false);
  const messagesEndRef = React.useRef(null);

  const activeChat = chats.find((chat) => chat.id === activeChatId) ?? chats[0];

  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages, isLoading]);

  function updateActiveChat(updater) {
    setChats((currentChats) =>
      currentChats.map((chat) =>
        chat.id === activeChatId ? { ...chat, ...updater(chat) } : chat
      )
    );
  }

  function handleNewChat() {
    const nextId = chatCounter + 1;
    const nextChat = createChat(nextId);
    setChatCounter(nextId);
    setChats((currentChats) => [nextChat, ...currentChats]);
    setActiveChatId(nextId);
    setDraft("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const message = draft.trim();
    if (!message || isLoading) return;

    const userMessage = { role: "user", content: message };
    const history = activeChat.messages;

    updateActiveChat((chat) => ({
      title: chat.title.startsWith("Chat ") ? message.slice(0, 34) || chat.title : chat.title,
      messages: [...chat.messages, userMessage]
    }));
    setDraft("");
    setIsLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history })
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "The assistant could not respond.");
      }

      updateActiveChat((chat) => ({
        messages: [...chat.messages, { role: "assistant", content: payload.answer }]
      }));
    } catch (error) {
      updateActiveChat((chat) => ({
        messages: [
          ...chat.messages,
          {
            role: "assistant",
            content: `I could not complete that request. ${error.message}`
          }
        ]
      }));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Search size={20} />
          </div>
          <div>
            <strong>FinSearch</strong>
            <span>Research assistant</span>
          </div>
        </div>

        <button className="new-chat" type="button" onClick={handleNewChat}>
          <MessageCirclePlus size={18} />
          <span>New Chat</span>
        </button>

        <div className="chat-list">
          {chats.map((chat) => (
            <button
              className={`chat-tab ${chat.id === activeChatId ? "active" : ""}`}
              key={chat.id}
              type="button"
              onClick={() => setActiveChatId(chat.id)}
              title={chat.title}
            >
              <PanelLeft size={16} />
              <span>{chat.title}</span>
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-area">
        <header className="topbar">
          <div>
            <p className="eyebrow">LlamaIndex powered search</p>
            <h1>Financial Research Chat</h1>
          </div>
          <div className="status-pill">
            <Sparkles size={16} />
            <span>Context memory on</span>
          </div>
        </header>

        <section className="messages" aria-live="polite">
          {activeChat.messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <div className="avatar">{message.role === "assistant" ? "AI" : "You"}</div>
              <div className="bubble">
                <MessageContent message={message} />
              </div>
            </article>
          ))}
          {isLoading && (
            <article className="message assistant">
              <div className="avatar">AI</div>
              <div className="bubble typing">
                <span />
                <span />
                <span />
              </div>
            </article>
          )}
          <div ref={messagesEndRef} />
        </section>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            aria-label="Chat prompt"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                handleSubmit(event);
              }
            }}
            placeholder="Ask about stocks, the report, trends, or market news"
            rows={1}
          />
          <button type="submit" disabled={!draft.trim() || isLoading} aria-label="Send message">
            <Send size={20} />
          </button>
        </form>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
