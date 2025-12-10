def divider(label: str = "") -> None:
    line = "─" * 40
    if label:
        print(f"\n{line}  {label.upper()}  {line}\n")
    else:
        print(f"\n{line * 2}\n")
