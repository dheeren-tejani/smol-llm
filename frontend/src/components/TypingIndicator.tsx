export function TypingIndicator() {
  return (
    <div className="flex items-center gap-2 animate-message-in">
      <div className="w-7 h-7 rounded-full bg-muted flex items-center justify-center text-xs font-bold text-foreground shrink-0">
        S
      </div>
      <div className="flex items-center gap-1 px-3 py-2">
        <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground typing-dot" />
        <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground typing-dot" />
        <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground typing-dot" />
      </div>
    </div>
  );
}
