CARGO_FLAGS += -j 1
CARGO_FLAGS += --release
# CARGO_FLAGS += --verbose

nergal: assets
	cargo build $(CARGO_FLAGS)

run: assets
	RUST_BACKTRACE=1 cargo run $(CARGO_FLAGS)

assets:
	# TODO: git clone --depth=1 https://github.com/ozkriff/nergal_assets assets
	mkdir assets

android: assets
	cargo apk

.PHONY: nirgal run android
