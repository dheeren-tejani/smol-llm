import type { CSSProperties, FC, ReactNode } from 'react';

export interface IconProps {
  size?: number;
  sw?: number;
  className?: string;
  style?: CSSProperties;
}

function svgIcon(paths: ReactNode, name: string): FC<IconProps> {
  const C: FC<IconProps> = function (props: IconProps) {
    return (
      <svg width={props.size ?? 16} height={props.size ?? 16} viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth={props.sw ?? 2} strokeLinecap="round" strokeLinejoin="round"
        className={props.className} style={props.style} aria-hidden="true">{paths}</svg>
    );
  };
  C.displayName = name;
  return C;
}

export const PanelLeftIcon = svgIcon(
  <><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M9 3v18" /></>, 'PanelLeft');

export const PlusIcon = svgIcon(
  <><path d="M5 12h14" /><path d="M12 5v14" /></>, 'Plus');

export const TrashIcon = svgIcon(
  <><path d="M3 6h18" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" /><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /><path d="M10 11v6" /><path d="M14 11v6" /></>, 'Trash');

export const InfoIcon = svgIcon(
  <><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></>, 'Info');

export const CheckIcon = svgIcon(
  <path d="M20 6 9 17l-5-5" />, 'Check');

export const CopyIcon = svgIcon(
  <><rect width="14" height="14" x="8" y="8" rx="2" ry="2" /><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" /></>, 'Copy');

export const XIcon = svgIcon(
  <><path d="M18 6 6 18" /><path d="m6 6 12 12" /></>, 'X');

export const ArrowUpIcon = svgIcon(
  <><path d="m5 12 7-7 7 7" /><path d="M12 19V5" /></>, 'ArrowUp');

export const ArrowDownIcon = svgIcon(
  <><path d="M12 5v14" /><path d="m19 12-7 7-7-7" /></>, 'ArrowDown');

export const RotateIcon = svgIcon(
  <><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /></>, 'RotateCcw');

export const AlertIcon = svgIcon(
  <><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" /><path d="M12 9v4" /><path d="M12 17h.01" /></>, 'AlertTriangle');

export const NetworkIcon = svgIcon(
  <><rect x="16" y="16" width="6" height="6" rx="1" /><rect x="2" y="16" width="6" height="6" rx="1" /><rect x="9" y="2" width="6" height="6" rx="1" /><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3" /><path d="M12 12V8" /></>, 'Network');

export const BracesIcon = svgIcon(
  <><path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1" /><path d="M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1" /></>, 'Braces');

export const TableIcon = svgIcon(
  <><rect width="18" height="18" x="3" y="3" rx="2" /><path d="M12 3v18" /><path d="M3 9h18" /><path d="M3 15h18" /></>, 'Table');

export const FileIcon = svgIcon(
  <><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" /><path d="M14 2v4a2 2 0 0 0 2 2h4" /><path d="M16 13H8" /><path d="M16 17H8" /><path d="M10 9H8" /></>, 'FileText');

export function LogoMark({ size, glow }: { size: number; glow?: boolean }) {
  const s = Math.round(size * 0.5);
  return (
    <div className={'logo-mark' + (glow ? ' logo-glow' : '')}
      style={{ width: size, height: size, borderRadius: Math.round(size * 0.3) }}>
      <svg width={s} height={s} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M12 2l2.4 7.6L22 12l-7.6 2.4L12 22l-2.4-7.6L2 12l7.6-2.4z" />
      </svg>
    </div>
  );
}