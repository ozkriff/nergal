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

android_install_run_log: android
	adb install -r $(APK)
	adb logcat -c
	adb shell am start -n rust.nergal/rust.nergal.MainActivity
	adb logcat | grep Rust

.PHONY: nirgal run android
