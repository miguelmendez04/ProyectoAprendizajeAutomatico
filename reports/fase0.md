# Fase 0 — Verificación de fuentes (2026-11-05)

## Hallazgos automatizados (API Socrata de datos.gov.co)
Los IDs de la guía (`m2nt-jw2h`, `w7bi-rwg5`, `86ha-bzcy`) **no existen** (HTTP 404 `dataset.missing`). Se reemplazan por:

| Fuente | ID verificado | Notas |
|---|---|---|
| Saber 11 (unificado) | `kgxf-xxbe` | 51 columnas, periodos 20101–20224 en una sola tabla; municipio del colegio `cole_cod_mcpio_ubicacion`, `punt_global`, `fami_estratovivienda`, `cole_naturaleza`. Periodos con sufijo 1/2/4 (calendario). |
| Matrícula ES por municipio | `y9ga-zwzy` | 19.618 filas; 2005 →; columnas por nivel de formación; `c_digo_delmunicipio` (5 dígitos), `a_o`. |
| Matrícula ES por IES×programa | `5wck-szir` | 390.903 filas, 2015–2021 (semestres 1 y 2); trae códigos IES, programa y municipio (códigos sin cero a la izquierda, ej. `5001`). |
| DIVIPOLA municipios | `gdxc-w37w` | 1.122 municipios; `cod_mpio` ya con 5 dígitos. |

- No hay datos abiertos de deserción por IES/municipio en datos.gov.co (solo deserción interna de IES individuales): **el objetivo viene solo de SPADIES**.
- Ventana temporal elegida: **2015–2021** (intersección de matrícula SNIES; Saber 11 cubre 2010–2022 para el rezago t-1).

## Pendiente (manual, requiere al usuario)
- [ ] (a) Consulta pública de SPADIES: indicadores, filtros, formatos de exportación (con capturas).
- [ ] (b) Probar si la cuenta institucional UAO da acceso ampliado.
- [ ] (d) Histórico de cambios de códigos municipales (DIVIPOLA vigente ya está en `gdxc-w37w`).
- [ ] TerriData (DNP): descarga manual desde terridata.dnp.gov.co.
