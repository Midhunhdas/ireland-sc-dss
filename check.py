with open('app.py') as f:
    lines = f.readlines()
print('Total lines:', len(lines))
found = any('sector-store' in l for l in lines)
print('Has sector-store:', found)
print('First line:', lines[0].strip())
print('Last line:', lines[-1].strip())
