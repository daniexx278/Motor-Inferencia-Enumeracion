import itertools

# ==========================================
# CLASES BASE
# ==========================================

class Node:
    def __init__(self, name, values):
        self.name = name
        self.values = [v.strip() for v in values.split(",")]
        self.parents = []
        self.children = []
        self.cpt = {}  # diccionario: clave=(padres,val) -> prob

    def add_parents(self, parents):
        self.parents = [p.strip() for p in parents.split(",")] if parents else []

    def set_cpt(self, entries):
        for e in entries:
            parts = e.split()
            if len(parts) == 2:  # sin padres
                val, prob = parts
                self.cpt[("none", val)] = float(prob)
            else:
                parent_info = " ".join(parts[:-2])
                val, prob = parts[-2:]
                self.cpt[(parent_info.strip(), val.strip())] = float(prob)

    def get_prob(self, value, evidence):
        """
        Devuelve P(this_node=value | padres en evidence)
        """
        if not self.parents:
            key = ("none", value)
        else:
            conds = []
            for p in self.parents:
                if p not in evidence:
                    raise ValueError(f"Falta valor de evidencia para padre {p}")
                conds.append(f"{p}={evidence[p]}")
            key = (" ".join(conds), value)

        if key not in self.cpt:
            raise ValueError(f"CPT no encontrado para {self.name} con clave {key}")
        return self.cpt[key]


class BayesNet:
    def __init__(self):
        self.nodes = {}

    def add_node(self, name, values):
        if name not in self.nodes:
            self.nodes[name] = Node(name, values)

    def add_edge(self, parent, child):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError(f"Nodo inexistente en arista {parent}->{child}")
        self.nodes[parent].children.append(child)
        self.nodes[child].parents.append(parent)

    def get_node(self, name):
        return self.nodes.get(name)

    def variables(self):
        return list(self.nodes.keys())

    def probability(self, var, value, evidence):
        node = self.get_node(var)
        return node.get_prob(value, evidence)

# ==========================================
# PARSERS DE ARCHIVOS
# ==========================================

def leer_estructura(path):
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                edges.append((parts[0], parts[1]))
    return edges


def leer_cpts(path):
    nodos = {}
    with open(path, "r") as f:
        contenido = f.read().split("Node:")
        for bloque in contenido:
            bloque = bloque.strip()
            if not bloque:
                continue
            lineas = bloque.split("\n")
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
            nodo.add_parents(padres)
            nodo.set_cpt(cpt)
            nodos[nombre] = nodo
    return nodos


def construir_red(edges_path, cpts_path):
    nodos = leer_cpts(cpts_path)
    bn = BayesNet()
    for name, node in nodos.items():
        bn.add_node(name, ",".join(node.values))
        bn.nodes[name] = node
    edges = leer_estructura(edges_path)
    for parent, child in edges:
        bn.add_edge(parent, child)
    return bn

# ==========================================
# MOTOR DE INFERENCIA (ENUMERACIÓN)
# ==========================================

def enumeration_ask(X, e, bn):
    Q = {}
    for x in bn.get_node(X).values:
        Q[x] = enumerate_all(bn.variables(), {**e, X: x}, bn)
    return normalizar(Q)


def enumerate_all(vars, e, bn):
    if not vars:
        return 1.0
    Y = vars[0]
    rest = vars[1:]
    node = bn.get_node(Y)

    if Y in e:
        prob = bn.probability(Y, e[Y], e)
        return prob * enumerate_all(rest, e, bn)
    else:
        total = 0
        for y in node.values:
            prob = bn.probability(Y, y, e)
            total += prob * enumerate_all(rest, {**e, Y: y}, bn)
        return total


def normalizar(Q):
    total = sum(Q.values())
    return {k: v / total for k, v in Q.items()}

# ==========================================
# MENÚ INTERACTIVO
# ==========================================

def mostrar_menu():
    print("\n==============================")
    print("  MOTOR DE INFERENCIA - IA")
    print("==============================")
    print("1. Cargar archivos de red (edges.txt, cpts.txt)")
    print("2. Mostrar estructura de la red")
    print("3. Ejecutar consulta manual")
    print("4. Cargar archivo de pruebas")
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
    X = input("Variable de consulta (ej: Appointment): ").strip()
    evid_str = input("Evidencias (ej: Rain=light,Maintenance=no): ").strip()
    evidencias = {}
    if evid_str:
        for par in evid_str.split(","):
            k, v = par.split("=")
            evidencias[k.strip()] = v.strip()

    resultado = enumeration_ask(X, evidencias, bn)
    print("\nResultados de inferencia:")
    for val, prob in resultado.items():
        print(f"P({X}={val} | evidencia) = {prob:.5f}")

def ejecutar_archivo_pruebas(bn):
    # Pedir al usuario el nombre del archivo de pruebas
    nombre = input("Ingrese el nombre del archivo de pruebas (ej: tren.txt): ").strip()
    if not nombre:
        print("⚠️  No ingresó ningún nombre de archivo.")
        return

    # Construir ruta completa dentro de la carpeta tests
    path = f"tests/{nombre}"

    # Intentar abrir el archivo
    try:
        with open(path, "r") as f:
            lineas = f.readlines()
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo: {path}")
        return

    print(f"\n✅ Leyendo pruebas desde {path}...\n")

    # Procesar cada línea del archivo de pruebas
    for i, line in enumerate(lineas):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("QUERY"):
            X = line.split()[1]

        elif line.startswith("EVIDENCE"):
            evid_str = line.replace("EVIDENCE", "").strip()
            evidencias = {}
            for par in evid_str.replace(":", "").split(","):
                k, v = par.split("=")
                evidencias[k.strip()] = v.strip()

            print(f"\n🔍 Prueba {i}: P({X} | {evidencias})")
            res = enumeration_ask(X, evidencias, bn)
            for val, prob in res.items():
                print(f"   {val}: {prob:.5f}")
            print("------------------------------")

# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def main():
    bn = None
    while True:
        mostrar_menu()
        op = input("Seleccione una opción: ").strip()
        if op == "1":
            bn = construir_red("data/edges.txt", "data/cpts.txt")
            print("✅ Archivos cargados correctamente.")
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
