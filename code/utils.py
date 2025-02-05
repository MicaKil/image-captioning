import pickle


def dump(obj: any, path: str):
	"""
	Dump an object to a file.
	:param obj: The object to dump
	:param path: Path to the file where the object will be dumped
	:return:
	"""
	with open(path, "wb") as f:
		pickle.dump(obj, f)


def load(path: str) -> any:
	"""
	Load an object from a file.
	:param path: Path to the file where the object is stored
	:return: The object loaded from the file
	"""
	with open(path, "rb") as f:
		return pickle.load(f)
