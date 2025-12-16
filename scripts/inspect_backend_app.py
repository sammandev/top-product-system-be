import importlib

m = importlib.import_module("backend_fastapi")
try:
    module_file = m.__file__
except AttributeError:
    module_file = None
print("backend_fastapi module file=", module_file)

try:
    all_names = m.__all__
except AttributeError:
    all_names = dir(m)
print("names in backend_fastapi:", list(all_names)[:50])

try:
    a = m.app

    print("type(backend_fastapi.app)=", type(a))
    print("repr=", repr(a))
    try:
        print("is callable?", callable(a))
        if hasattr(a, "__class__"):
            print("class name", a.__class__)
    except Exception as e:
        print("inspect failed", e)
except Exception as e:
    print("could not get app attr:", e)

# Also inspect backend_fastapi.app submodule
try:
    sub = importlib.import_module("backend_fastapi.app")
    try:
        sub_file = sub.__file__
    except AttributeError:
        sub_file = None
    print("backend_fastapi.app module file=", sub_file)
    print("sub module attrs sample:", [n for n in dir(sub) if not n.startswith("_")][:50])
    try:
        sub_app = sub.app
    except AttributeError:
        sub_app = None
    print("sub.app attr repr:", sub_app)
except Exception as e:
    print("could not import backend_fastapi.app", e)
