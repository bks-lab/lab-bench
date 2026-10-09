/**
 * Raw chat prompts for the local arms: one system turn, one user turn, the
 * opening of the assistant turn, then `prefix`. Thinking is switched off
 * where the family has a thinking mode. Rule for adding a family
 * (plan/t12.md): copy the turn markers of the template Ollama ships with the
 * model (`/api/show`, field `template`).
 *
 *   chatml   Qwen3, Qwen3.8: empty think block (Qwen3's non-thinking form)
 *   gemma4   Gemma 4, Winnow-12B, shisa-de-1: empty thought channel
 *   gemma    Gemma 2/3: system line inside the user turn
 *   phi4     Phi-4 (no thinking mode)
 *   mistral  Mistral Small 3.2 (no thinking mode)
 *   granite  Granite 4 (no thinking mode)
 */
export const TEMPLATES = ['chatml', 'gemma4', 'gemma', 'phi4', 'mistral', 'granite'];

export function rawPrompt(template, system, user, prefix = '') {
  switch (template) {
    case 'gemma4': return `<|turn>system\n${system}<turn|>\n<|turn>user\n${user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>${prefix}`;
    case 'gemma': return `<start_of_turn>user\n${system}\n\n${user}<end_of_turn>\n<start_of_turn>model\n${prefix}`;
    case 'phi4': return `<|im_start|>system<|im_sep|>\n${system}<|im_end|>\n<|im_start|>user<|im_sep|>\n${user}<|im_end|>\n<|im_start|>assistant<|im_sep|>\n${prefix}`;
    case 'mistral': return `[SYSTEM_PROMPT]${system}[/SYSTEM_PROMPT][INST]${user}[/INST]${prefix}`;
    case 'granite': return `<|start_of_role|>system<|end_of_role|>${system}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>${user}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>${prefix}`;
    case 'chatml': return `<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n${prefix}`;
    default: throw new Error(`unknown template ${template}`);
  }
}
