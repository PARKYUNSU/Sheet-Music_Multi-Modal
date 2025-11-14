import sounddevice as sd
def list_devices():
    info = sd.query_devices()
    inputs = []
    outputs = []
    for i, d in enumerate(info):
        name = d.get('name', f'device_{i}')
        if d.get('max_input_channels', 0) > 0:
            inputs.append({'index': i, 'name': name})
        if d.get('max_output_channels', 0) > 0:
            outputs.append({'index': i, 'name': name})
    return inputs, outputs
if __name__ == '__main__':
    ins, outs = list_devices()
    print('=== Input devices ===')
    for d in ins: print(f"{d['index']:>3} : {d['name']}")
    print('\n=== Output devices ===')
    for d in outs: print(f"{d['index']:>3} : {d['name']}")
