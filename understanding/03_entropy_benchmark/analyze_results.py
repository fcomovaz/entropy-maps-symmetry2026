"""Generate compact measured tables from CSVs, without rerunning timings."""
import csv
import math
import json
import hashlib
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
R = ROOT/'results'


def read(name):
    with (R/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def med(rows,key='median_ns'):
    return statistics.median(float(r[key]) for r in rows)


def main():
    entropy,pipeline,hist,luts,errors,ranks = [read(n+'.csv') for n in
        ('entropy_benchmark','pipeline_benchmark','histogram_benchmark','lut_memory_build','errors','ranking')]
    assert len(entropy)==480 and len(pipeline)==1200 and len(hist)==300 and len(luts)==8
    assert all(float(r['max_absolute_error']) <= 1e-12 for r in errors if r['method'] in ('E2','E3','E4'))
    assert all(0 <= float(r['top_k_agreement']) <= 1 for r in ranks)
    metadata = json.loads((R/'run_metadata.json').read_text(encoding='utf-8'))
    # Keep the original execution hash; path-only relocations have a separate audit record.
    expected_hash = metadata.get('relocation', {}).get('script_sha256', metadata['script_sha256'])
    assert expected_hash==hashlib.sha256((ROOT/'entropy_benchmark.py').read_bytes()).hexdigest()
    reproducible = None
    if (R/'run1/patch_scores.csv').exists():
        reproducible = all((R/name).read_bytes()==(R/'run1'/name).read_bytes()
                           for name in ('manifest.csv','patch_scores.csv','errors.csv','ranking.csv'))
        assert reproducible, 'Deterministic data or numerical outputs changed between full runs'
    (R/'validation.json').write_text(json.dumps(dict(
        entropy_rows=len(entropy),pipeline_rows=len(pipeline),histogram_rows=len(hist),
        ranking_patches=len(read('patch_scores.csv')),exact_comparisons=metadata['exact_comparisons'],
        exact_tolerance_pass=True,script_hash_matches=True,
        script_hash_basis='relocation' if 'relocation' in metadata else 'execution',
        two_run_data_errors_scores_ranking_byte_identical=reproducible),indent=2)+'\n',encoding='utf-8')
    lines = ['# Resultados medidos','',
        'Medianas entre los casos de cada tamaño; cada caso ya es mediana de batches. '
        'No son intervalos de confianza ni un workload representativo del paper. '
        'Unidades ns/patch. Consultar CSV para distribución y dispersión.','',
        '| N | E1 | E2 | E3 | E4 | H1+E1 | H1+E4 | Mejor pipeline (mediana) |',
        '|---:|---:|---:|---:|---:|---:|---:|---|']
    for size in (16,32,64,128):
        e = [r for r in entropy if int(r['size'])==size]
        p = [r for r in pipeline if int(r['size'])==size]
        methods = sorted({r['method'] for r in p})
        winner = min(methods,key=lambda m:med([r for r in p if r['method']==m]))
        vals = [med([r for r in e if r['method']==m]) for m in ('E1','E2','E3','E4')]
        vals += [med([r for r in p if r['method']==m]) for m in ('H1+E1','H1+E4')]
        lines.append('| '+str(size*size)+' | '+' | '.join(f'{v:.0f}' for v in vals)+f' | {winner}: {med([r for r in p if r["method"]==winner]):.0f} |')
    lines += ['','## Histograma aislado','','| Tamaño | H1 | H2 array | H5 unroll4 | H3 Counter | E5 tracking |','|---:|---:|---:|---:|---:|---:|']
    for size in (16,32,64,128):
        vals = [med([r for r in hist if int(r['size'])==size and r['method']==m]) for m in ('H1','H2_array','H5_unroll4','H3_Counter','H1_sparse')]
        lines.append('| '+str(size)+' | '+' | '.join(f'{v:.0f}' for v in vals)+' |')
    lines += ['','## E5 y fracción de entropía','','Cada ratio compara pipelines medidos: >1 favorece tracking. '
              'Fracciones estimadas con componentes aislados; overhead/caché aparecen como residual en CSV.','',
              '| Tamaño | Caso | H1+E4 / E5+E4 | H1+E1 / E5+E1 | Fracción H1+E1 entropía | Fracción H1+E4 entropía |',
              '|---:|---|---:|---:|---:|---:|']
    for size in (16,32,64,128):
        for group in ('single_level','few_levels','uniform_256','random','real'):
            rows = [r for r in pipeline if int(r['size'])==size and r['group']==group]
            by = {m:[r for r in rows if r['method']==m] for m in ('H1+E1','H1+E4','H1_sparse+E1','H1_sparse+E4')}
            lines.append(f'| {size} | {group} | {med(by["H1+E4"])/med(by["H1_sparse+E4"]):.3f} | '
                f'{med(by["H1+E1"])/med(by["H1_sparse+E1"]):.3f} | '
                f'{med(by["H1+E1"],"estimated_entropy_fraction"):.1%} | {med(by["H1+E4"],"estimated_entropy_fraction"):.1%} |')
    lines += ['','## LUT: construcción y amortización','','Amortización frente a E1 = build_ns / (mediana E1 − mediana LUT). '
              'Es una estimación dependiente de estos casos, sin carga desde ROM.','',
              '| N | LUT | Entradas | C f64 bytes | C f32/u32 bytes | C u16 bytes | Python bytes | Build ns | Patches para amortizar |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in luts:
        e = [x for x in entropy if x['size']==r['size']]
        saving = med([x for x in e if x['method']=='E1'])-med([x for x in e if x['method']==r['method']])
        breakeven = str(math.ceil(float(r['median_ns'])/saving)) if saving>0 else 'no'
        lines.append('| '+' | '.join(r[k] for k in ('N','method','entries','c_float64_bytes','c_float32_bytes','c_uint16_bytes','python_deep_bytes'))+f' | {float(r["median_ns"]):.0f} | {breakeven} |')
    lines += ['','## Error y ranking','','Grupo all, pares estrictos con tolerancia 1e-12. '
              'Posiciones y top-k usan orden descendente sin tolerancia y desempate por índice, '
              'por eso pueden variar incluso con 0 inversiones estrictas.','',
              '| Tamaño | Método | MAE | Máx abs | Máx relativo (H>1e-12) | Inversiones | Empates nuevos | Empates rotos | Posiciones distintas | Top-1 | Top-10 |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for er in errors:
        if er['group']!='all':
            continue
        rr = [r for r in ranks if r['group']=='all' and r['size']==er['size'] and r['method']==er['method']]
        first = rr[0]
        top = {r['k']:r['top_k_agreement'] for r in rr}
        lines.append(f'| {er["size"]} | {er["method"]} | {float(er["mean_absolute_error"]):.3g} | {float(er["max_absolute_error"]):.3g} | {float(er["max_relative_error"]):.3g} | '+
            ' | '.join(first[k] for k in ('strict_inversions','collapsed_pairs','broken_reference_ties','position_discrepancies'))+f' | {top["1"]} | {top["10"]} |')
    lines += ['','## Top-k: casos reales y casi empatados','',
              '| Tamaño | Grupo | Método | Inversiones | Mínimo agreement entre k evaluados |',
              '|---:|---|---|---:|---:|']
    for size in (16,32,64,128):
        for group in ('real','near_tie'):
            for method in ('E3','E4','Q16','Q30'):
                rr = [r for r in ranks if int(r['size'])==size and r['group']==group and r['method']==method]
                lines.append(f'| {size} | {group} | {method} | {rr[0]["strict_inversions"]} | {min(float(r["top_k_agreement"]) for r in rr):.4f} |')
    if (R/'run1/pipeline_benchmark.csv').exists():
        old = read('run1/pipeline_benchmark.csv')
        lines += ['','## Repetición completa','',
            'Run1 se conserva sin sobrescribir. Cambió la resolución de la ruta del experimento 01, '
            'sin cambiar kernels ni bytes de entrada. La primera ejecución coincidió con actividad externa '
            'en el workspace. Las diferencias muestran variabilidad del host, no aceleraciones del algoritmo.','',
            '| Tamaño | Pipeline | Run1 ns | Run2 ns | Cambio |','|---:|---|---:|---:|---:|']
        for size in (16,32,64,128):
            for method in ('H1+E1','H1+E4','H3_Counter+E4','H1_sparse+E4'):
                a = med([r for r in old if int(r['size'])==size and r['method']==method])
                b = med([r for r in pipeline if int(r['size'])==size and r['method']==method])
                lines.append(f'| {size} | {method} | {a:.0f} | {b:.0f} | {b/a-1:+.1%} |')
    (R/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(R/'summary.md')


if __name__=='__main__':
    main()
