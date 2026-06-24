# ensight-reader-rs

A pure Rust reader for the EnSight Gold file format.

This library provides access to:

* EnSight Gold case files (`.case`)
* Geometry files
* Per-node scalar variables
* Per-node vector variables
* Time-dependent datasets
* Multi-part geometries
* Unstructured meshes

The project was developed primarily for high-performance scientific data processing workflows involving large CFD datasets.

## Features

* Pure Rust implementation
* No Python dependency
* Supports binary EnSight Gold files
* Supports time series datasets
* Supports all common EnSight element types
* Efficient random access using file offsets
* Suitable for HPC and large-scale post-processing workflows

## Supported Element Types

* point
* bar2
* bar3
* tria3
* tria6
* quad4
* quad8
* tetra4
* tetra10
* pyramid5
* pyramid13
* penta6
* penta15
* hexa8
* hexa20
* nsided
* nfaced

## Installation

### Git dependency

```toml
[dependencies]
ensight-reader-rs = { git = "https://github.com/therealfatdoge/ensight-reader-rs.git" }
```

### Version tag

```toml
[dependencies]
ensight-reader-rs = { git = "https://github.com/therealfatdoge/ensight-reader-rs.git", tag = "v0.1.0" }
```

## Example

Open a case file:

```rust
use ensight_reader_rs::read_case;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let case = read_case("run_12n.case")?;

    println!("Variables:");

    for name in case.variables.keys() {
        println!("  {}", name);
    }

    Ok(())
}
```

Read geometry:

```rust
use ensight_reader_rs::read_case;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let case = read_case("run_12n.case")?;

    let geometry = case.get_geometry_model(0)?;

    for part in geometry.iter_parts() {
        println!(
            "Part {}: {} ({} nodes)",
            part.part_id,
            part.part_name,
            part.number_of_nodes
        );
    }

    Ok(())
}
```

Read variable data:

```rust
use ensight_reader_rs::read_case;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let case = read_case("run_12n.case")?;

    let variable = case.get_variable("nox", 0)?;

    let mut file = variable.open()?;

    if let Some(values) = variable.read_node_data(&mut file, 1)? {
        println!(
            "Read {} values",
            values.rows * values.cols
        );
    }

    Ok(())
}
```

## Intended Use Cases

* CFD post-processing
* Scientific visualization pipelines
* HPC data processing
* Data conversion to Parquet, ORC, Arrow, HDF5, etc.
* Large-scale simulation analytics

## Limitations

Current implementation supports:

* Binary EnSight Gold files
* Per-node variables

Not yet implemented:

* Per-element variables
* Writing EnSight files
* Parallel I/O
* Compressed EnSight files

## Credits

This project was inspired by and partially based on the design and functionality of the Python `ensight-reader` package.

The original project can be found at:

https://github.com/tkarabela/ensight-reader

See `CREDITS.md` for detailed attribution information.

## License

This project is licensed under the terms specified in the LICENSE file.

Please ensure compatibility with the license of any upstream projects referenced in `CREDITS.md`.
