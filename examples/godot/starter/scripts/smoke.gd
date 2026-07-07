extends SceneTree

func _init() -> void:
	var runtime := XybridRuntime.new()
	var result := runtime.init("", "", "", OS.get_user_data_dir().path_join("xybrid-cache"))
	if result.ok:
		print("xybrid godot smoke ok")
		quit(0)
	else:
		push_error(result.message)
		quit(1)
