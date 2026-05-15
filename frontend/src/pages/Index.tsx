import { useState } from 'react';
import { useChatState } from '@/hooks/useChatState';
import { ChatHeader } from '@/components/ChatHeader';
import { ChatSidebar } from '@/components/ChatSidebar';
import { ChatArea } from '@/components/ChatArea';
import { ChatInput } from '@/components/ChatInput';

const Index = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const {
    messages,
    parameters,
    setParameters,
    isLoading,
    inputValue,
    setInputValue,
    sendMessage,
    clearChat,
  } = useChatState();

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">
      <ChatHeader
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(prev => !prev)}
        onNewChat={clearChat}
      />
      <div className="flex flex-1 overflow-hidden">
        <ChatSidebar
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          parameters={parameters}
          onParametersChange={setParameters}
        />
        <div className="flex-1 flex flex-col min-w-0">
          <ChatArea messages={messages} isLoading={isLoading} onSuggestion={(text) => sendMessage(text)} />
          <ChatInput
            value={inputValue}
            onChange={setInputValue}
            onSend={() => sendMessage(inputValue)}
            disabled={isLoading}
          />
        </div>
      </div>
    </div>
  );
};

export default Index;
