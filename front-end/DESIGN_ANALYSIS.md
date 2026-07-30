# AI Chatbot Interface v2: component analysis

Reference: https://dribbble.com/shots/27040052-AI-Chatbot-Interface-Design-v2

## Visual system

- Near-black canvas with layered dark-green surfaces.
- Muted emerald as the primary interaction and AI-state color.
- Soft neutral text hierarchy rather than pure white everywhere.
- Compact sans typography for navigation and controls; large low-tracking display type for the greeting.
- Medium corner radius across cards, inputs, buttons and panels.
- Thin low-contrast borders separate modules without heavy shadows.
- Gradients and glows are used only around the AI workspace, not on every component.

## Application shell

- A fixed left sidebar holds identity, navigation, recent conversations, data sources and account/model status.
- A sticky top bar identifies the active workspace and exposes evidence mode plus settings.
- The main canvas is intentionally sparse before the first prompt.
- A bottom-fixed composer remains reachable throughout the conversation.

## Sidebar components

- Brand mark and product wordmark.
- New-research action with keyboard shortcut.
- Primary navigation rows with icon, label and selected state.
- Recent-thread section with compact title and category metadata.
- Collapsible data-source panel with checkbox states.
- Model/API status tile with semantic connection indicator.
- Mobile close control and scrim.

## Workspace components

- Compact AI identity marker above the greeting.
- Two-line greeting headline.
- Scope-setting description.
- Four quick-agent cards. Each has an icon tile, title, capability line and launch affordance.
- Empty state is the welcome composition itself, avoiding an additional generic empty card.

## Composer components

- Auto-growing prompt textarea.
- Tools action.
- Search-mode action.
- Primary send control.
- Inline validation error.
- Trust note that reminds users to verify incomplete metadata.
- Enter submits; Shift+Enter creates a line break; Ctrl/Cmd+K focuses the composer.

## Result-conversation components

- User prompt bubble.
- Assistant identity row and execution status.
- Skeleton thinking state shaped like the eventual response.
- Explanation of verified and unverified evidence lanes.
- Source and sensitive-domain notices.
- Expandable agent trace.
- Verified dataset rows with source, link, reasoning, metadata, fit score and four sub-scores.
- Unverified web rows with explicit warning and two preliminary scores.
- Cross-source and potential-duplicate notices.
- Empty lane states.
- Fallback guidance block.
- Contextual error response inside the conversation.

## Responsive and motion behavior

- Desktop uses a persistent 292px sidebar.
- Mobile converts the sidebar into a dismissible drawer with scrim.
- Four agent cards collapse to two columns and then one column.
- Dataset score grids collapse from four columns to two.
- Motion supports hierarchy and state change: initial reveal, card hover lift, assistant response entrance and thinking pulse.
- All nonessential motion is disabled with `prefers-reduced-motion`.
