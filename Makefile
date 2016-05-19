CARGO_FLAGS += -j 1
CARGO_FLAGS += --release
# CARGO_FLAGS += --verbose

nergal: assets
	cargo build $(CARGO_FLAGS)

run: assets
	RUST_BACKTRACE=1 cargo run $(CARGO_FLAGS)

assets:
	git clone --depth=1 https://github.com/ozkriff/nergal_assets assets

APK = ./target/android-artifacts/build/bin/nergal-debug.apk

android: assets
	cargo apk
	adb install -r $(APK)
	adb logcat -c
	adb logcat | grep Rust

.PHONY: nirgal run android
