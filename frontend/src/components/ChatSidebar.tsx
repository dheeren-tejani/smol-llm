import { X } from 'lucide-react';
import { ChatParameters } from '@/lib/types';
import { ParameterControl } from './ParameterControl';

interface ChatSidebarProps {
  open: boolean;
  onClose: () => void;
  parameters: ChatParameters;
  onParametersChange: (params: ChatParameters) => void;
}

export function ChatSidebar({ open, onClose, parameters, onParametersChange }: ChatSidebarProps) {
  const updateParam = (key: keyof ChatParameters, value: number) => {
    onParametersChange({ ...parameters, [key]: value });
  };

  return (
    <>
      {/* Mobile overlay */}
      {open && (
        <div
          className="fixed inset-0 bg-background/10 z-30 md:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`
          fixed md:relative z-40 top-0 left-0 h-full
          bg-secondary 
          transition-all duration-300 ease-in-out
          overflow-hidden
          ${open ? 'w-[280px]' : 'w-0'}
        `}
      >
        <div className="w-[280px] h-full flex flex-col overflow-y-auto">
          {/* Close on mobile */}
          <div className="flex items-center justify-end p-3 md:hidden">
            <button
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-muted transition-colors duration-200"
            >
              <X className="w-4 h-4 text-muted-foreground" />
            </button>
          </div>

          {/* About */}
          <div className="px-4 py-6">
            <p className="text-sm text-muted-foreground leading-relaxed">
              <span className="text-foreground font-medium">Dheeren's Chat</span> is a portfolio project developed by Dheeren. Trained the model from scratch with cosmopedia, performed SFT, created both backend and frontend server with UI for chatting, feel free to test it.
            </p>
            <p className="text-sm text-muted-foreground mt-3">
              GitHub:{' '}
              <a      
                href="https://github.com/dheeren-tejani/smol-llm"
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground underline underline-offset-2 hover:text-accent transition-colors duration-200"
              >
                dheeren-tejani/smol-llm
              </a>
            </p>
          </div>

          {/* Divider */}
          <div className="mx-4 border-t border-border" />

          {/* Parameters */}
          <div className="px-4 py-3 flex-1">
            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">Parameters</h3>
            <div className="space-y-5">
              <ParameterControl
                label="Max Output Tokens"
                description="Maximum number of tokens the model will generate in a single response."
                value={parameters.max_tokens}
                min={1}
                max={1024}
                step={1}
                onChange={v => updateParam('max_tokens', v)}
                type="number"
              />
              <ParameterControl
                label="Temperature"
                description="Controls randomness. Lower = more focused, higher = more creative."
                value={parameters.temperature}
                min={0}
                max={2}
                step={0.01}
                onChange={v => updateParam('temperature', v)}
                type="slider"
              />
              <ParameterControl
                label="Top P"
                description="Nucleus sampling threshold. The model samples from the top P probability mass."
                value={parameters.top_p}
                min={0}
                max={1}
                step={0.01}
                onChange={v => updateParam('top_p', v)}
                type="slider"
              />
              <ParameterControl
                label="Top K"
                description="Limits sampling to the top K most likely tokens at each step."
                value={parameters.top_k}
                min={1}
                max={200}
                step={1}
                onChange={v => updateParam('top_k', v)}
                type="number"
              />
              <ParameterControl
                label="Repetition Penalty"
                description="Penalizes tokens that have already appeared. Reduces repetitive output."
                value={parameters.repetition_penalty}
                min={1}
                max={2}
                step={0.01}
                onChange={v => updateParam('repetition_penalty', v)}
                type="slider"
              />
              <ParameterControl
                label="Range Epsilon"
                description="Minimum probability threshold for token filtering. Decides how many logits are most relevant to the prompt."
                value={0.2}
                min={0}
                max={1}
                step={0.01}
                onChange={v => updateParam('range_epsilon', v)}
                type="number"
              />
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
