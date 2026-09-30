import sys
import types


class _Event:
    def __init__(self, fn, inputs, outputs):
        self.fn = fn
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.then_events = []

    def then(self, fn=None, inputs=None, outputs=None, **_kwargs):
        event = _Event(fn, inputs, outputs)
        self.then_events.append(event)
        return event


class _Component:
    def __init__(self, stub, kind, *args, **kwargs):
        self.stub = stub
        self.kind = kind
        self.args = args
        self.label = kwargs.get("label")
        self.value = kwargs.get("value")
        self.choices = kwargs.get("choices")
        self.visible = kwargs.get("visible", True)
        self.parent = stub.stack[-1] if stub.stack else None
        self.events = []
        stub.components.append(self)

    def __enter__(self):
        self.stub.stack.append(self)
        return self

    def __exit__(self, *_args):
        self.stub.stack.pop()

    def _bind(self, fn=None, inputs=None, outputs=None, **_kwargs):
        event = _Event(fn, inputs, outputs)
        self.events.append(event)
        return event

    click = _bind
    change = _bind


class _GradioStub(types.ModuleType):
    def __init__(self):
        super().__init__("gradio")
        self.components = []
        self.stack = []

    def __getattr__(self, kind):
        return lambda *args, **kwargs: _Component(self, kind, *args, **kwargs)

    @staticmethod
    def update(**kwargs):
        return kwargs

    @staticmethod
    def Warning(_message):
        return None


def _construct_webui(monkeypatch):
    import webui

    gradio = _GradioStub()
    monkeypatch.setitem(sys.modules, "gradio", gradio)
    webui.build_ui()
    return webui, gradio


def _named(components, label):
    return [component for component in components if component.label == label]


def _studio(gradio):
    return next(
        component
        for component in gradio.components
        if component.kind == "Accordion" and component.args == ("VoxCPM2 Voice Studio",)
    )


def _studio_controls(gradio):
    studio = _studio(gradio)
    controls = []
    for component in gradio.components:
        parent = component.parent
        while parent is not None and parent is not studio:
            parent = parent.parent
        if parent is studio:
            controls.append(component)
    return controls


def _is_effectively_visible(component):
    current = component
    while current is not None:
        if current.visible is False:
            return False
        current = current.parent
    return True


def _apply_result(event, result):
    for component, update in zip(event.outputs, result):
        if isinstance(update, dict):
            for name, value in update.items():
                setattr(component, name, value)


def test_webui_exports_import_and_voxcpm_helpers():
    import webui

    assert webui.CLI_LANG == "en"
    assert "Local File" in webui.MEDIA_SOURCE_CHOICES
    assert "Direct Download URL" in webui.MEDIA_SOURCE_CHOICES
    assert "YouTube" in webui.MEDIA_SOURCE_CHOICES
    assert callable(webui.import_media_for_webui)
    assert callable(webui.test_voxcpm2_voice_for_webui)
    assert callable(webui.use_voxcpm2_voice_for_webui)


def test_english_option_labels_preserve_chinese_language_support():
    import webui

    assert "No subtitles" in webui.SUBTITLE_TYPES
    assert "Hard subtitles" in webui.SUBTITLE_TYPES
    assert any("中文" in str(name) or "Chinese" in str(name) for name in webui.LANG_DISPLAY_NAMES)


def test_constructed_webui_has_one_voxcpm2_voice_profile_selector(monkeypatch):
    webui, gradio = _construct_webui(monkeypatch)

    profile_choices = tuple(webui.profile_names())
    selectors = [
        component
        for component in gradio.components
        if component.kind == "Dropdown"
        and (component.label == "Voice Profile" or tuple(component.choices or ()) == profile_choices)
    ]

    assert len(selectors) == 1


def test_constructed_webui_has_one_reusable_generate_test_voice_control(monkeypatch):
    _webui, gradio = _construct_webui(monkeypatch)

    button_values = [component.args[0] for component in gradio.components if component.kind == "Button" and component.args]
    audition_controls = [
        value
        for value in button_values
        if value in {"Generate Test Voice", "Test Voice", "Test Voice Again"}
    ]

    assert audition_controls == ["Generate Test Voice"]


def test_voxcpm2_studio_starts_hidden_for_a_non_voxcpm_provider(monkeypatch):
    _webui, gradio = _construct_webui(monkeypatch)

    controls = _studio_controls(gradio)

    assert controls
    assert all(not _is_effectively_visible(control) for control in controls)


def test_tts_provider_change_gates_every_voxcpm2_studio_control(monkeypatch):
    webui, gradio = _construct_webui(monkeypatch)
    monkeypatch.setattr(webui.tts, "is_input_api", lambda **_kwargs: True)
    monkeypatch.setattr(webui, "role_menu", lambda *_args, **_kwargs: ["voice"])
    tts_provider = _named(gradio.components, "TTS Provider")[0]
    change_event = tts_provider.events[0]
    voxcpm2 = next(name for name in webui.TTS_NAMES if "VoxCPM" in name)
    edge_tts = next(name for name in webui.TTS_NAMES if "Edge" in name)
    controls = _studio_controls(gradio)

    hidden_result = change_event.fn(edge_tts, voxcpm2, "English")
    _apply_result(change_event, hidden_result)
    controls_hidden = all(not _is_effectively_visible(control) for control in controls)
    visible_result = change_event.fn(voxcpm2, edge_tts, "English")
    _apply_result(change_event, visible_result)
    controls_visible = all(_is_effectively_visible(control) for control in controls)

    assert (controls_hidden, controls_visible) == (True, True)


def test_voxcpm2_edge_voxcpm2_transition_restores_saved_voice(monkeypatch):
    webui, gradio = _construct_webui(monkeypatch)
    monkeypatch.setattr(webui.tts, "is_input_api", lambda **_kwargs: True)
    tts_provider = _named(gradio.components, "TTS Provider")[0]
    change_event = tts_provider.events[0]
    voxcpm2 = next(name for name in webui.TTS_NAMES if "VoxCPM" in name)
    edge_tts = next(name for name in webui.TTS_NAMES if "Edge" in name)
    saved_voice = webui.profile_names()[-1]
    voxcpm2_index = webui._tts_index_from_display(voxcpm2)

    def roles(tts_index, **_kwargs):
        return webui.profile_names() if tts_index == voxcpm2_index else ["edge-voice"]

    monkeypatch.setattr(webui, "role_menu", roles)

    def invoke(provider, previous_provider):
        values = []
        for component in change_event.inputs:
            if component is tts_provider:
                values.append(provider)
            elif component.label == "Target Language":
                values.append("English")
            elif component.label == "Voice Profile":
                values.append(saved_voice)
            else:
                values.append(previous_provider)
        return change_event.fn(*values)

    edge_result = invoke(edge_tts, voxcpm2)
    restored_result = invoke(voxcpm2, edge_tts)
    voice_role_position = change_event.outputs.index(_named(gradio.components, "Voice Profile")[0])

    assert edge_result[voice_role_position]["value"] == "edge-voice"
    assert restored_result[voice_role_position]["value"] == saved_voice
