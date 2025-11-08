# motor_inferencia_enu.py
# Versión corregida: evita duplicados en padres/hijos y asegura el orden de padres según CPT.

import itertools

# ==========================================
# CLASES BASE
# ==========================================

class Node:
    def __init__(self, name, values):
        self.name = name
        self.values = [v.strip() for v in values.split(",")] if values else []
        self.parents = []            # lista de nombres de padres (orden usada para construir claves CPT)
        self.children = []          # lista de nombres de hijos
        self.cpt = {}               # diccionario: clave=(parent_condition_string, value) -> prob
        self.expected_parents = []  # listado de padres según el bloque CPT (mantiene orden deseado)

    def add_expected_parents(self, parents):
        """Guardar la lista de padres esperada (según cpts.txt) sin sobrescribir parents aún."""
        if not parents:
            self.expected_parents = []
        else:
            self.expected_parents = [p.strip() for p in parents.split(",")]

    def set_cpt(self, entries):
        """
        Recibe una lista de líneas CPT y las guarda en el diccionario.
        Para nodos sin padres, la clave es ("none", valor)
        Para nodos con padres, la clave es ("Padre1=val1 Padre2=val2", valor)
        """
        for e in entries:
            parts = e.split()
            if len(parts) == 2:
                val, prob = parts
                key = ("none", val.strip())
                self.cpt[key] = float(prob)
            else:
                # asumimos que los últimos dos tokens son "valor prob"
                val, prob = parts[-2], parts[-1]
                parent_info = " ".join(parts[:-2]).strip()
                key = (parent_info, val.strip())
                self.cpt[key] = float(prob)

    def get_prob(self, value, evidence):
        """
        Devuelve P(this_node = value | padres en evidence).
        Construye la clave en el mismo orden que self.parents.
        """
        if not self.parents:
            key = ("none", value)
        else:
            conds = []
            for p in self.parents:
                if p not in evidence:
                    raise ValueError(f"Falta valor de evidencia para padre '{p}' del nodo '{self.name}'")
                conds.append(f"{p}={evidence[p]}")
            key = (" ".join(conds), value)

        if key not in self.cpt:
            # Mostrar mensaje con info útil para depuración
            raise ValueError(f"CPT no encontrado para {self.name} con clave {key}. "
                             f"Claves disponibles: {list(self.cpt.keys())}")
        return self.cpt[key]


class BayesNet:
    def __init__(self):
        self.nodes = {}

    def add_node(self, name, values=""):
        if name not in self.nodes:
            self.nodes[name] = Node(name, values)

    def add_edge(self, parent, child):
        """
        Añade la arista parent -> child sin duplicar entradas en listas.
        """
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError(f"Nodo inexistente en arista {parent}->{child}")
        if child not in self.nodes[parent].children:
            self.nodes[parent].children.append(child)
        if parent not in self.nodes[child].parents:
            self.nodes[child].parents.append(parent)

    def get_node(self, name):
        return self.nodes.get(name)

    def variables(self):
        # Devuelve una lista consistente de variables (orden de inserción)
        return list(self.nodes.keys())

    def probability(self, var, value, evidence):
        node = self.get_node(var)
        return node.get_prob(value, evidence)


# ==========================================
# PARSERS DE ARCHIVOS
# ==========================================

def leer_estructura(path):
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                edges.append((parts[0], parts[1]))
    return edges


def leer_cpts(path):
    """
    Lee el archivo cpts.txt y devuelve un dict nombre->Node (con CPT cargada y expected_parents).
    NOTA: no setea aún node.parents porque queremos conservar el orden definido en 'Parents:'.
    """
    nodos = {}
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read().split("Node:")
        for bloque in contenido:
            bloque = bloque.strip()
            if not bloque:
                continue
            lineas = [l for l in bloque.split("\n") if l.strip() != ""]
            nombre = lineas[0].strip()
            valores = ""
            padres = ""
            cpt = []
            for line in lineas[1:]:
                line = line.strip()
                if line.startswith("Values:"):
                    valores = line.replace("Values:", "").strip()
                elif line.startswith("Parents:"):
                    padres = line.replace("Parents:", "").strip()
                elif line and not line.startswith("CPT"):
                    cpt.append(line)
            nodo = Node(nombre, valores)
            # Guardamos el orden esperado de padres según cpts.txt
            nodo.add_expected_parents(padres)
            nodo.set_cpt(cpt)
            nodos[nombre] = nodo
    return nodos


def construir_red(edges_path, cpts_path):
    # Leemos nodos desde CPTs (esto da el listado y el orden esperado de padres)
    nodos_from_cpt = leer_cpts(cpts_path)
    bn = BayesNet()

    # Añadir nodos al bn
    for name, node in nodos_from_cpt.items():
        bn.add_node(name, ",".join(node.values))
        # reemplazamos el Node vacío por el que vino del parser (con CPT y expected_parents)
        bn.nodes[name] = node

    # Leer aristas y agregarlas (sin duplicar)
    edges = leer_estructura(edges_path)
    for parent, child in edges:
        # Asegurar los nodos existen
        if parent not in bn.nodes:
            raise ValueError(f"Parent '{parent}' en edges.txt no existe en cpts.txt")
        if child not in bn.nodes:
            raise ValueError(f"Child '{child}' en edges.txt no existe en cpts.txt")
        # Añadir relación
        if child not in bn.nodes[parent].children:
            bn.nodes[parent].children.append(child)
        if parent not in bn.nodes[child].parents:
            bn.nodes[child].parents.append(parent)

    # IMPORTANTE: Ajustar el orden de parents de cada nodo según expected_parents (si existe)
    for name, node in bn.nodes.items():
        if node.expected_parents:
            # Reasignar parents en el orden que indica el CPT
            node.parents = [p for p in node.expected_parents if p]  # limpia valores vacíos
            # Asegurar que cada padre liste a este nodo como hijo
            for p in node.parents:
                if name not in bn.nodes[p].children:
                    bn.nodes[p].children.append(name)

    # Eliminar posibles duplicados finales (precaución)
    for name, node in bn.nodes.items():
        node.parents = list(dict.fromkeys(node.parents))
        node.children = list(dict.fromkeys(node.children))

    return bn


# ==========================================
# MOTOR DE INFERENCIA (ENUMERACIÓN)
# ==========================================

def enumeration_ask(X, e, bn):
    """
    Implementación de enumeration-ask de Russell & Norvig.
    Devuelve dict valor->probabilidad normalizada P(X | e)
    """
    if X not in bn.nodes:
        raise ValueError(f"Variable de consulta {X} no encontrada en la red.")
    Q = {}
    for x in bn.get_node(X).values:
        Q[x] = enumerate_all(bn.variables(), {**e, X: x}, bn)
    return normalizar(Q)


def enumerate_all(vars, e, bn):
    """
    vars: lista de nombres de variables (en orden consistente)
    e: dict de evidencia parcial
    """
    if not vars:
        return 1.0
    Y = vars[0]
    rest = vars[1:]
    node = bn.get_node(Y)

    if Y in e:
        prob = bn.probability(Y, e[Y], e)
        return prob * enumerate_all(rest, e, bn)
    else:
        total = 0.0
        for y in node.values:
            prob = bn.probability(Y, y, e)
            total += prob * enumerate_all(rest, {**e, Y: y}, bn)
        return total


def normalizar(Q):
    total = sum(Q.values())
    if total == 0:
        return {k: 0 for k in Q}
    return {k: v / total for k, v in Q.items()}


# ==========================================
# MENÚ INTERACTIVO Y UTILIDADES
# ==========================================

def mostrar_menu():
    print("\n==============================")
    print("  MOTOR DE INFERENCIA - IA")
    print("==============================")
    print("1. Cargar archivos de red (data/edges.txt, data/cpts.txt)")
    print("2. Mostrar estructura de la red")
    print("3. Ejecutar consulta manual")
    print("4. Cargar archivo de pruebas (tests/<nombre>)")
    print("5. Salir")
    print("==============================")


def mostrar_red(bn):
    print("\nEstructura de la Red Bayesiana:")
    for name, node in bn.nodes.items():
        print(f"- {name}: padres={node.parents}, hijos={node.children}")
    print("\nTablas de Probabilidad (CPTs):")
    for name, node in bn.nodes.items():
        print(f"\nNodo: {name}")
        for key, prob in node.cpt.items():
            print(f"  {key}: {prob}")


def ejecutar_consulta_manual(bn):
    try:
        X = input("Variable de consulta (ej: Appointment): ").strip()
        evid_str = input("Evidencias (ej: Rain=light,Maintenance=no): ").strip()
        evidencias = {}
        if evid_str:
            for par in evid_str.split(","):
                if "=" not in par:
                    print(f"Formato inválido en evidencia: '{par}'. Debe ser clave=valor.")
                    return
                k, v = par.split("=")
                evidencias[k.strip()] = v.strip()

        resultado = enumeration_ask(X, evidencias, bn)
        print("\nResultados de inferencia:")
        for val, prob in resultado.items():
            print(f"P({X}={val} | evidencia) = {prob:.5f}")
    except Exception as ex:
        print("Error en la consulta manual:", ex)


def ejecutar_archivo_pruebas(bn):
    nombre = input("Ingrese el nombre del archivo de pruebas (ej: tren.txt): ").strip()
    if not nombre:
        print("⚠️  No ingresó ningún nombre de archivo.")
        return

    # Comprobar distintas rutas posibles
    paths_to_try = [f"tests/{nombre}", nombre]
    contenido = None
    for path in paths_to_try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                contenido = f.read()
            used_path = path
            break
        except FileNotFoundError:
            continue

    if contenido is None:
        print(f"❌ No se encontró el archivo: ninguno de {paths_to_try}")
        return

    print(f"\n✅ Leyendo pruebas desde {used_path}...\n")
    lineas = contenido.splitlines()
    X = None
    for i, raw in enumerate(lineas, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("QUERY"):
            parts = line.split()
            if len(parts) >= 2:
                X = parts[1].strip()
            else:
                print(f"Línea {i}: QUERY sin variable.")
                X = None
        elif line.upper().startswith("EVIDENCE"):
            if X is None:
                print(f"Línea {i}: EVIDENCE sin QUERY previo. Ignorando.")
                continue
            evid_str = line[len("EVIDENCE"):].strip()
            evidencias = {}
            if evid_str:
                for par in evid_str.replace(":", "").split(","):
                    if "=" not in par:
                        print(f"Línea {i}: formato inválido en evidencia '{par}'")
                        continue
                    k, v = par.split("=")
                    evidencias[k.strip()] = v.strip()
            try:
                print(f"\n🔍 Prueba (línea {i}): P({X} | {evidencias})")
                res = enumeration_ask(X, evidencias, bn)
                for val, prob in res.items():
                    print(f"   {val}: {prob:.5f}")
                print("------------------------------")
            except Exception as ex:
                print(f"Error al evaluar prueba (línea {i}): {ex}")


# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def main():
    bn = None
    while True:
        mostrar_menu()
        op = input("Seleccione una opción: ").strip()
        if op == "1":
            try:
                bn = construir_red("data/edges.txt", "data/cpts.txt")
                print("✅ Archivos cargados correctamente.")
            except Exception as ex:
                print("Error al cargar archivos:", ex)
        elif op == "2":
            if bn:
                mostrar_red(bn)
            else:
                print("⚠️  Primero cargue los archivos (opción 1).")
        elif op == "3":
            if bn:
                ejecutar_consulta_manual(bn)
            else:
                print("⚠️  Primero cargue los archivos (opción 1).")
        elif op == "4":
            if bn:
                ejecutar_archivo_pruebas(bn)
            else:
                print("⚠️  Primero cargue los archivos (opción 1).")
        elif op == "5":
            print("👋 Saliendo del sistema. ¡Hasta luego!")
            break
        else:
            print("Opción no válida, intente de nuevo.")


if __name__ == "__main__":
    main()
