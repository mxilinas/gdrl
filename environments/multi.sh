#!/bin/sh
echo -ne '\033c\033]0;godot-gdrl\a'
base_path="$(dirname "$(realpath "$0")")"
"$base_path/multi.x86_64" "$@"
