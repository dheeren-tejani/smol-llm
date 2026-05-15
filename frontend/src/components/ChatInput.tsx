import { useRef, useEffect } from 'react';
import { Send, Plus } from 'lucide-react';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled: boolean;
  maxLength?: number;
}

export function ChatInput({ value, onChange, onSend, disabled, maxLength = 1024 }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 160) + 'px';
    }
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !disabled) onSend();
    }
  };

  return (
    <div className="shrink-0 bg-background px-4 py-3">
      <div className="max-w-2xl mx-auto">
        <div className="flex items-end gap-2 bg-card border border-border rounded-xl px-3 py-2 transition-shadow duration-200 focus-within:shadow-[0_0_0_1px_hsl(0_0%_20%)]">
          <button
            className="w-8 h-8 flex items-center justify-center rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors duration-200 shrink-0 mb-0.5"
            aria-label="Attach file"
          >
            <Plus className="w-4 h-4" />
          </button>

          <textarea
            ref={textareaRef}
            value={value}
            onChange={e => {
              if (e.target.value.length <= maxLength) onChange(e.target.value);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Message...."
            disabled={disabled}
            rows={1}
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none resize-none min-h-[36px] max-h-[160px] py-1.5 disabled:opacity-50"
          />

          <button
            onClick={onSend}
            disabled={!value.trim() || disabled}
            className={`w-8 h-8 flex items-center justify-center rounded-md transition-all duration-200 shrink-0 mb-0.5 ${
              value.trim() && !disabled
                ? 'bg-primary text-primary-foreground opacity-100'
                : 'opacity-0 pointer-events-none'
            }`}
            aria-label="Send message"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        <div className="flex justify-end mt-1">
          <span className="text-[10px] text-muted-foreground tabular-nums">
            {value.length}/{maxLength}
          </span>
        </div>
      </div>
    </div>
  );
}
