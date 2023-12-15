
all_specs = {
    "circle": {
        "input_size": (16,16),
        "output_size": (256,256),
        "inputs": ["lowres"],
        "output": "hires",
        "pattern": "circle"
    },
    "tidal_circle1": {
        "input_size": (6,6),
        "output_size": (256,256),
        "inputs": ["lowres","tide_3d"],
        "output": "hires",
        "pattern": "tidal_circle",
        "hyperparameters": {
            "fc_size":32,
            "encoded_dim_size":8,
            "nr_epochs": 2000
        }
    },
    "curve1": {
        "input_size": (16,16),
        "output_size": (256,256),
        "inputs": ["lowres","tide_3d"],
        "output": "hires",
        "pattern": "curve"
    },
    "circle2": {
        "input_size": (24,20),
        "output_size": (280,256),
        "inputs": ["lowres"],
        "output": "hires",
        "pattern": "circle"
    }
}