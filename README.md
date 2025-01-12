# Actividad M1

## Propósito

Este repositorio contiene la implementación de una simulación de robots de limpieza para la actividad del curso TC2008B. El objetivo es aplicar herramientas para la implementación de sistemas multiagente.

## Detalles de la Simulación

### Dado:

- **Dimensiones de la Habitación**: Una cuadrícula de celdas.
- **Número de Agentes**: Total de robots de limpieza.
- **Porcentaje de Celdas Sucias Iniciales**.
- **Tiempo Máximo de Ejecución**.

### Pasos:

1. **Inicialización**: Celdas sucias asignadas aleatoriamente.
2. **Inicio de Agentes**: Todos comienzan en la celda [1,1].
3. **Comportamiento**:
   - Limpian si la celda está sucia.
   - Se mueven aleatoriamente si está limpia.
4. **Terminación**: Cuando todas las celdas están limpias o se alcanza el tiempo límite.

## Recolección de Datos

- Tiempo necesario para limpiar.
- Porcentaje de celdas limpias al final.
- Número total de movimientos.

## Especificaciones de los Agentes

- Cada integrante diseña un tipo o configuración de agente.
- Se evalúa el rendimiento en diferentes corridas (25%, 50%, 75%, 100% del tiempo máximo).
- Identificación del agente óptimo.
