"""Configuration management for Project Vertex v4."""
import json
import os


class Config:
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.config_file = config_file
        self._defaults = {
            "display":    {"width": 1200, "height": 900, "fov": 45, "near": 0.1, "far": 50.0},
            "camera":     {"initial_z": -5, "zoom_min": -15, "zoom_max": -2},
            "controls":   {"rotation_sensitivity": 0.5, "smoothing_factor": 0.12},
            "rendering":  {"wireframe": False, "show_grid": True, "show_axes": False,
                           "show_hud": True, "shape_scale": 1.0},
            "hand_sensor":{"detection_confidence": 0.7, "tracking_confidence": 0.7,
                           "min_palm_px": 35},
            "scene":      {"default": "arc_reactor", "startup_demo": False},
            "voice":      {"enabled": True, "wake_word": "", "audio_feedback": False,
                           "whisper_model": "tiny.en"},
        }
        self.config = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file) as f:
                    user = json.load(f)
                return self._merge(self._defaults, user)
            except Exception as e:
                print(f">> Config load error: {e}. Using defaults.")
        else:
            self._save(self._defaults)
        return dict(self._defaults)

    def _save(self, cfg: dict):
        try:
            with open(self.config_file, "w") as f:
                json.dump(cfg, f, indent=4)
        except Exception as e:
            print(f">> Config save error: {e}")

    def _merge(self, base: dict, override: dict) -> dict:
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._merge(result[k], v)
            else:
                result[k] = v
        return result

    def get(self, *keys):
        val = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return None

    def set(self, *keys, value):
        cfg = self.config
        for k in keys[:-1]:
            cfg = cfg.setdefault(k, {})
        cfg[keys[-1]] = value
        self._save(self.config)

    def save(self):
        self._save(self.config)
