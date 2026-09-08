from inference import load_artifacts
model, labels, config = load_artifacts()
print("Model ready:", config["selected_model"], "| Categories:", len(labels))
