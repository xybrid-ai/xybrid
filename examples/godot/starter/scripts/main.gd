extends Control

@onready var output: Label = $Output

func _ready() -> void:
	var runtime := XybridRuntime.new()
	var cache_dir := OS.get_user_data_dir().path_join("xybrid-cache")
	var init_result := runtime.init("", "", "", cache_dir)
	if not init_result.ok:
		output.text = init_result.message
		return

	var model_id := OS.get_environment("XYBRID_GODOT_MODEL")
	if model_id.is_empty():
		output.text = "Xybrid initialized. Set XYBRID_GODOT_MODEL to run a registry model."
		return

	var loader_result := runtime.model_from_registry(model_id)
	if not loader_result.ok:
		output.text = loader_result.message
		return

	var async := XybridAsync.new()
	async.completed.connect(_on_model_loaded)
	async.load_model(loader_result.value)
	output.text = "Loading %s..." % model_id

func _on_model_loaded(result: Dictionary) -> void:
	if not result.ok:
		output.text = result.message
		return

	var model: XybridModel = result.value
	var run_result := model.run({"kind": "text", "text": "Hello from Godot"}, {}, {})
	if run_result.ok:
		output.text = run_result.value.get("text", str(run_result.value))
	else:
		output.text = run_result.message
