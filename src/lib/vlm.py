import base64
import io
import re
import numpy as np
from PIL import Image

PROMPT_TEMPLATE_CUSTOM = (
    "You are a STRICT judge evaluating whether a robotic manipulation task was completed "
    "in a generated video. Your DEFAULT verdict is FAILURE -- only output SUCCESS when you "
    "can clearly verify the instructed change by directly analyzing and comparing the "
    "INITIAL, INTERMEDIATE (if applicable) and FINAL frames.\n\n"
    "Task instruction: \"{instruction}\"\n\n"
    "You are given {n_frames} frames in temporal order: the first is the INITIAL state, "
    "the last is the FINAL state, and any in-between images are intermediate states. "
    "The frames may contain minor visual artifacts from the video model -- judge "
    "TASK SEMANTICS, not image quality.\n\n"
    "Reason step-by-step. Keep each step to one short sentence :\n"
    "1. NAMED OBJECT: identify the exact object named in the instruction (color + shape).\n"
    "2. TARGET: state the target (another object, a location, or a side) that the named "
    "object must move toward / next to / onto. If occluded in the initial frame, use the earliest visible frame as reference.\n"
    "3. INITIAL POSITION: where is the named object relative to the target initially?\n"
    "4. FINAL POSITION: where is the named object relative to the target in the LAST frame?\n"
    "5. DISPLACEMENT: did the named object visibly displace between FIRST and LAST frame? "
    "If the position is essentially unchanged, the verdict is FAILURE.\n"
    "6. DIRECTION: if it moved, did it end up CLOSER to the target than it started? If it "
    "moved in the wrong direction or away from the target, the verdict is FAILURE.\n\n"
    "OCCLUSION HANDLING (important):\n"
    "- The robot arm is the white/grey gripper that often hangs in the scene. If the NAMED "
    "OBJECT is HIDDEN by the robot arm or by another object in the INITIAL frame, do NOT "
    "conclude that it was missing or 'introduced later'. Use the earliest frame where the "
    "object becomes visible as the initial reference. If the named object is clearly at the "
    "correct position in the FINAL frame, the verdict can still be SUCCESS even when you "
    "could not see it in frame 0.\n"
    "- Similarly, if the named object is occluded ONLY in the FINAL frame, judge on the latest visible frame.\n\n"
    "Strict rules (apply without exception):\n"
    "- If a DIFFERENT object moved instead of the one named, output FAILURE.\n"
    "- If the named object did not visibly move closer to the target than it started, output FAILURE.\n"
    "- If you cannot tell whether the named object moved, output FAILURE -- uncertainty is "
    "NOT a reason to give SUCCESS.\n"
    "- Do NOT fabricate the final position.\n"
    "- Minor blurriness/artifacts from the video model are acceptable -- judge TASK SEMANTICS, "
    "not image quality. But \"I cannot tell whether it moved\" is NOT an excuse for SUCCESS; "
    "it is FAILURE.\n\n"
    "After the reasoning, output a FINAL line containing ONLY one word: SUCCESS or FAILURE."
)

def parse_verdict_with_status(text):
    verdict_match = re.search(r"VERDICT\s*:\s*(SUCCESS|FAILURE)", text, flags=re.IGNORECASE)
    if verdict_match:
        return verdict_match.group(1).upper() == "SUCCESS", "verdict_line"

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        tail = re.sub(r"[^A-Za-z]", "", lines[-1]).upper()
        if tail == "SUCCESS":
            return True, "tail_line"
        if tail == "FAILURE":
            return False, "tail_line"

    upper = text.upper()
    last_s = max((m.end() for m in re.finditer(r"\bSUCCESS\b", upper)), default=-1)
    last_f = max((m.end() for m in re.finditer(r"\bFAILURE\b", upper)), default=-1)
    if last_s == -1 and last_f == -1:
        return False, "default_no_signal"
    return last_s > last_f, "keyword_search"

def _pil_to_api_image(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

def _sample_indices(total, n):
    if total <= 0:
        return []
    if n >= total:
        return list(range(total))
    if n == 1:
        return [total - 1]
    return np.linspace(0, total - 1, n).round().astype(int).tolist()


def judge_video(client, model, frames, instruction):
    selected = [frames[i] for i in _sample_indices(len(frames), 6)]
    images = [Image.fromarray((frame.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)) for frame in selected]
    prompt = PROMPT_TEMPLATE_CUSTOM.format(instruction=instruction, n_frames=len(images))
    content = [{"type": "text", "text": prompt}]
    content += [_pil_to_api_image(image) for image in images]
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}],
                                              temperature=0.0, max_tokens=768)
    text = response.choices[0].message.content or ""
    success, status = parse_verdict_with_status(text)
    return {"success": success, "parse_status": status, "response": text}
