class_name XybridAsync
extends RefCounted

signal completed(result: Dictionary)

var _thread: Thread

func is_running() -> bool:
	return _thread != null and _thread.is_alive()

func load_model(loader: XybridModelLoader) -> void:
	_start(Callable(loader, "load"))

func run_model(model: XybridModel, envelope: Dictionary, generation_config := {}, run_options := {}) -> void:
	_start(Callable(model, "run").bind(envelope, generation_config, run_options))

func wait_to_finish() -> Dictionary:
	if _thread == null:
		return {"ok": false, "code": 0, "retryable": false, "message": "no worker thread"}

	var result: Dictionary = _thread.wait_to_finish()
	_thread = null
	return result

func _start(callable: Callable) -> void:
	if is_running():
		push_error("XybridAsync already has a running operation")
		return

	_thread = Thread.new()
	_thread.start(func():
		var result: Dictionary = callable.call()
		call_deferred("_finish", result)
		return result
	)

func _finish(result: Dictionary) -> void:
	if _thread != null:
		_thread.wait_to_finish()
		_thread = null
	completed.emit(result)
