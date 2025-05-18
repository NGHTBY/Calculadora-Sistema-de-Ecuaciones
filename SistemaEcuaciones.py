import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from fractions import Fraction
import numpy as np

class SistemaEcuacionesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Ecuaciones Lineales ++")
        self.root.geometry("1000x800")
        self.root.configure(bg='white')
        
        # Variables de control
        self.tamano_sistema = tk.IntVar(value=2)
        self.metodo = tk.StringVar(value="gauss")
        self.coeficientes = []
        self.terminos_independientes = []
        self.variables = ['x', 'y', 'z', 'w', 'v', 'u']
        self.historial = []
        
        # Crear interfaz
        self.crear_interfaz()

    def crear_interfaz(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame de configuración
        config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding="10")
        config_frame.pack(fill=tk.X, pady=5)
        
        # Controles de configuración
        ttk.Label(config_frame, text="Tamaño:").grid(row=0, column=0, padx=5)
        ttk.Combobox(config_frame, textvariable=self.tamano_sistema, 
                    values=[2, 3, 4, 5, 6], state="readonly").grid(row=0, column=1, padx=5)
        ttk.Button(config_frame, text="Crear Sistema", command=self.crear_sistema).grid(row=0, column=2, padx=5)
        
        # Botones adicionales
        ttk.Button(config_frame, text="Ayuda", command=self.mostrar_ayuda).grid(row=0, column=3, padx=5)
        ttk.Button(config_frame, text="Tema", command=self.cambiar_tema).grid(row=0, column=4, padx=5)
        ttk.Button(config_frame, text="Historial", command=self.mostrar_historial).grid(row=0, column=5, padx=5)
        
        # Frame de ecuaciones
        self.ecuaciones_frame = ttk.LabelFrame(main_frame, text="Ecuaciones", padding="10")
        self.ecuaciones_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Frame de métodos
        metodo_frame = ttk.Frame(main_frame)
        metodo_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(metodo_frame, text="Método:").grid(row=0, column=0, padx=5)
        self.metodo_combobox = ttk.Combobox(metodo_frame, textvariable=self.metodo, 
                                          values=["gauss", "gauss-jordan", "sustitucion", "igualacion", "eliminacion"],
                                          state="readonly")
        self.metodo_combobox.grid(row=0, column=1, padx=5)
        self.metodo_combobox.bind("<<ComboboxSelected>>", self.actualizar_metodo)
        
        # Botones de acción
        self.resolver_btn = tk.Button(metodo_frame, text="Resolver", fg="white", bg="green",
                                    command=self.resolver_sistema, font=('Arial', 10, 'bold'))
        self.resolver_btn.grid(row=0, column=2, padx=5)
        
        self.limpiar_btn = tk.Button(metodo_frame, text="Limpiar", fg="white", bg="red",
                                   command=self.limpiar_sistema, font=('Arial', 10, 'bold'))
        self.limpiar_btn.grid(row=0, column=3, padx=5)
        
        # Área de resultados
        self.resultado_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=100, height=20,
                                                      foreground='black', background='white', font=('Arial', 10))
        self.resultado_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Crear sistema inicial
        self.crear_sistema()

    def mostrar_ayuda(self):
        ayuda_ventana = tk.Toplevel(self.root)
        ayuda_ventana.title("Ayuda")
        ayuda_ventana.geometry("500x300")
        
        texto_ayuda = """INSTRUCCIONES DE USO:

1. SELECCIONA EL TAMAÑO:
   - Elige el número de ecuaciones (de 2 a 6).

2. INGRESA LOS COEFICIENTES:
   - Escribe valores numéricos en cada casilla.
   - Usa fracciones (1/2) o decimales (0.5).

3. ELIGE UN MÉTODO:
   - Para 2x2: Sustitución, Igualación o Eliminación.
   - Para sistemas mayores: Gauss o Gauss-Jordan.

4. RESUELVE:
   - Haz clic en 'Resolver' para ver la solución.
   - Usa 'Limpiar' para reiniciar.

FUNCIONES ADICIONALES:
- Tema: Cambia entre modo claro/oscuro.
- Historial: Muestra sistemas y soluciones."""
        
        texto = tk.Text(ayuda_ventana, wrap=tk.WORD, padx=10, pady=10)
        texto.insert(tk.END, texto_ayuda)
        texto.config(state=tk.DISABLED)
        texto.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(ayuda_ventana, text="Cerrar", command=ayuda_ventana.destroy).pack(pady=5)

    def cambiar_tema(self):
        if self.root.cget('bg') == 'white':
            # Cambiar a oscuro
            self.root.configure(bg='#2d2d2d')
            self.resultado_text.configure(bg='#1e1e1e', fg='white')
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.LabelFrame):
                    widget.configure(style='Dark.TLabelframe')
        else:
            # Cambiar a claro
            self.root.configure(bg='white')
            self.resultado_text.configure(bg='white', fg='black')
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.LabelFrame):
                    widget.configure(style='TLabelframe')

    def actualizar_metodo(self, event=None):
        if self.tamano_sistema.get() == 2:
            self.metodo_combobox['values'] = ["sustitucion", "igualacion", "eliminacion", "gauss", "gauss-jordan"]
        else:
            self.metodo_combobox['values'] = ["gauss", "gauss-jordan"]

    def crear_sistema(self):
        # Limpiar frame de ecuaciones
        for widget in self.ecuaciones_frame.winfo_children():
            widget.destroy()
        
        n = self.tamano_sistema.get()
        self.coeficientes = []
        self.terminos_independientes = []
        
        # Crear entradas para cada ecuación
        for i in range(n):
            ecuacion_frame = ttk.Frame(self.ecuaciones_frame)
            ecuacion_frame.pack(fill=tk.X, pady=2)
            
            coeficientes_fila = []
            for j in range(n):
                entry = ttk.Entry(ecuacion_frame, width=5)
                entry.pack(side=tk.LEFT, padx=2)
                ttk.Label(ecuacion_frame, text=f"{self.variables[j]} +" if j < n-1 else f"{self.variables[j]} =").pack(side=tk.LEFT, padx=2)
                coeficientes_fila.append(entry)
            
            termino_entry = ttk.Entry(ecuacion_frame, width=5)
            termino_entry.pack(side=tk.LEFT, padx=2)
            
            self.coeficientes.append(coeficientes_fila)
            self.terminos_independientes.append(termino_entry)
        
        self.actualizar_metodo()

    def limpiar_sistema(self):
        self.crear_sistema()
        self.resultado_text.delete(1.0, tk.END)

    def obtener_matriz(self):
        n = self.tamano_sistema.get()
        matriz = []
        vector = []
        
        try:
            for i in range(n):
                fila = []
                for j in range(n):
                    valor = self.coeficientes[i][j].get()
                    if '/' in valor:
                        num, den = map(float, valor.split('/'))
                        fila.append(num / den)
                    else:
                        fila.append(float(valor))
                matriz.append(fila)
                
                termino = self.terminos_independientes[i].get()
                if '/' in termino:
                    num, den = map(float, termino.split('/'))
                    vector.append(num / den)
                else:
                    vector.append(float(termino))
            
            return np.array(matriz), np.array(vector)
        except ValueError:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos")
            return None, None

    def obtener_sistema_como_texto(self, matriz, vector):
        texto = ""
        n = len(matriz)
        for i in range(n):
            ecuacion = ""
            for j in range(n):
                ecuacion += f"{matriz[i][j]:.2f}{self.variables[j]} + "
            texto += ecuacion[:-3] + f" = {vector[i]:.2f}\n"
        return texto

    def resolver_sistema(self):
        matriz, vector = self.obtener_matriz()
        if matriz is None:
            return
        
        n = self.tamano_sistema.get()
        metodo = self.metodo.get()
        
        self.resultado_text.delete(1.0, tk.END)
        solucion_str = ""
        
        if n == 2 and metodo in ["sustitucion", "igualacion", "eliminacion"]:
            solucion_str = self.resolver_2x2(matriz, vector, metodo)
        else:
            if metodo == "gauss":
                solucion_str = self.resolver_gauss(matriz, vector)
            else:
                solucion_str = self.resolver_gauss_jordan(matriz, vector)
        
        # Registrar en historial
        self.historial.append({
            "tamano": n,
            "metodo": metodo,
            "sistema": self.obtener_sistema_como_texto(matriz, vector),
            "solucion": solucion_str
        })
        if len(self.historial) > 10:
            self.historial.pop(0)

    def resolver_2x2(self, matriz, vector, metodo):
        a, b = matriz[0][0], matriz[0][1]
        c, d = matriz[1][0], matriz[1][1]
        e, f = vector[0], vector[1]
        
        self.resultado_text.insert(tk.END, f"Sistema:\n{a}x + {b}y = {e}\n{c}x + {d}y = {f}\n\n")
        
        det = a * d - b * c
        if det != 0:
            self.resultado_text.insert(tk.END, "Solución única.\n\n")
        elif a/c == b/d == e/f:
            self.resultado_text.insert(tk.END, "Infinitas soluciones.\n\n")
            return self.resultado_text.get("1.0", tk.END)
        else:
            self.resultado_text.insert(tk.END, "Sin solución.\n\n")
            return self.resultado_text.get("1.0", tk.END)
        
        if metodo == "sustitucion":
            self.metodo_sustitucion_2x2(a, b, c, d, e, f)
        elif metodo == "igualacion":
            self.metodo_igualacion_2x2(a, b, c, d, e, f)
        elif metodo == "eliminacion":
            self.metodo_eliminacion_2x2(a, b, c, d, e, f)
        
        return self.resultado_text.get("1.0", tk.END)

    def metodo_sustitucion_2x2(self, a, b, c, d, e, f):
        self.resultado_text.insert(tk.END, "Método de sustitución:\n\n")
        
        # Despejar x en primera ecuación
        self.resultado_text.insert(tk.END, f"x = ({e} - {b}y)/{a}\n")
        
        # Sustituir en segunda ecuación
        self.resultado_text.insert(tk.END, f"{c}*(({e}-{b}y)/{a}) + {d}y = {f}\n")
        
        # Resolver para y
        coef_y = d - (c*b)/a
        term_indep = f - (c*e)/a
        y = term_indep / coef_y
        self.resultado_text.insert(tk.END, f"y = {y}\n")
        
        # Encontrar x
        x = (e - b*y)/a
        self.resultado_text.insert(tk.END, f"x = {x}\n\n")
        
        self.verificar_solucion_2x2(x, y, a, b, c, d, e, f)

    def metodo_igualacion_2x2(self, a, b, c, d, e, f):
        self.resultado_text.insert(tk.END, "Método de igualación:\n\n")
        
        # Despejar x en ambas
        self.resultado_text.insert(tk.END, f"x = ({e}-{b}y)/{a}\n")
        self.resultado_text.insert(tk.END, f"x = ({f}-{d}y)/{c}\n")
        
        # Igualar
        self.resultado_text.insert(tk.END, f"({e}-{b}y)/{a} = ({f}-{d}y)/{c}\n")
        
        # Resolver para y
        coef_y = a*d - c*b
        term_indep = a*f - c*e
        y = term_indep / coef_y
        self.resultado_text.insert(tk.END, f"y = {y}\n")
        
        # Encontrar x
        x = (e - b*y)/a
        self.resultado_text.insert(tk.END, f"x = {x}\n\n")
        
        self.verificar_solucion_2x2(x, y, a, b, c, d, e, f)

    def metodo_eliminacion_2x2(self, a, b, c, d, e, f):
        self.resultado_text.insert(tk.END, "Método de eliminación:\n\n")
        
        # Hacer coeficientes iguales
        self.resultado_text.insert(tk.END, f"Ecuación 1 * {c}: {a*c}x + {b*c}y = {e*c}\n")
        self.resultado_text.insert(tk.END, f"Ecuación 2 * {a}: {c*a}x + {d*a}y = {f*a}\n")
        
        # Restar ecuaciones
        coef_y = b*c - d*a
        term_indep = e*c - f*a
        self.resultado_text.insert(tk.END, f"{coef_y}y = {term_indep}\n")
        
        # Resolver para y
        y = term_indep / coef_y
        self.resultado_text.insert(tk.END, f"y = {y}\n")
        
        # Encontrar x
        x = (e - b*y)/a
        self.resultado_text.insert(tk.END, f"x = {x}\n\n")
        
        self.verificar_solucion_2x2(x, y, a, b, c, d, e, f)

    def verificar_solucion_2x2(self, x, y, a, b, c, d, e, f):
        self.resultado_text.insert(tk.END, "Verificación:\n")
        res1 = a*x + b*y
        res2 = c*x + d*y
        self.resultado_text.insert(tk.END, f"Ecuación 1: {res1} (debería ser {e})\n")
        self.resultado_text.insert(tk.END, f"Ecuación 2: {res2} (debería ser {f})\n")
        
        if abs(res1 - e) < 1e-6 and abs(res2 - f) < 1e-6:
            self.resultado_text.insert(tk.END, "\n¡Solución correcta!\n")

    def resolver_gauss(self, matriz, vector):
        n = len(matriz)
        self.resultado_text.insert(tk.END, "Eliminación Gaussiana:\n\n")
        
        aumentada = np.column_stack((matriz, vector))
        self.mostrar_matriz(aumentada, "Matriz inicial:")
        
        # Eliminación hacia adelante
        for i in range(n):
            if aumentada[i, i] == 0:
                for j in range(i+1, n):
                    if aumentada[j, i] != 0:
                        aumentada[[i, j]] = aumentada[[j, i]]
                        self.resultado_text.insert(tk.END, f"\nIntercambio fila {i+1} y {j+1}\n")
                        self.mostrar_matriz(aumentada)
                        break
            
            pivote = aumentada[i, i]
            if pivote != 1:
                aumentada[i] = aumentada[i] / pivote
                self.resultado_text.insert(tk.END, f"\nFila {i+1} ÷ {pivote}\n")
                self.mostrar_matriz(aumentada)
            
            for j in range(i+1, n):
                factor = aumentada[j, i]
                aumentada[j] -= factor * aumentada[i]
                self.resultado_text.insert(tk.END, f"\nFila {j+1} -= {factor}×Fila {i+1}\n")
                self.mostrar_matriz(aumentada)
        
        # Sustitución hacia atrás
        solucion = np.zeros(n)
        for i in range(n-1, -1, -1):
            solucion[i] = aumentada[i, -1]
            for j in range(i+1, n):
                solucion[i] -= aumentada[i, j] * solucion[j]
        
        self.resultado_text.insert(tk.END, "\nSolución:\n")
        for i in range(n):
            self.resultado_text.insert(tk.END, f"{self.variables[i]} = {solucion[i]}\n")
        
        self.verificar_solucion(matriz, vector, solucion)
        return self.resultado_text.get("1.0", tk.END)

    def resolver_gauss_jordan(self, matriz, vector):
        n = len(matriz)
        self.resultado_text.insert(tk.END, "Gauss-Jordan:\n\n")
        
        aumentada = np.column_stack((matriz, vector))
        self.mostrar_matriz(aumentada, "Matriz inicial:")
        
        for i in range(n):
            if aumentada[i, i] == 0:
                for j in range(i+1, n):
                    if aumentada[j, i] != 0:
                        aumentada[[i, j]] = aumentada[[j, i]]
                        self.resultado_text.insert(tk.END, f"\nIntercambio fila {i+1} y {j+1}\n")
                        self.mostrar_matriz(aumentada)
                        break
            
            pivote = aumentada[i, i]
            if pivote != 1:
                aumentada[i] = aumentada[i] / pivote
                self.resultado_text.insert(tk.END, f"\nFila {i+1} ÷ {pivote}\n")
                self.mostrar_matriz(aumentada)
            
            for j in range(n):
                if j != i:
                    factor = aumentada[j, i]
                    aumentada[j] -= factor * aumentada[i]
                    self.resultado_text.insert(tk.END, f"\nFila {j+1} -= {factor}×Fila {i+1}\n")
                    self.mostrar_matriz(aumentada)
        
        solucion = aumentada[:, -1]
        self.resultado_text.insert(tk.END, "\nSolución:\n")
        for i in range(n):
            self.resultado_text.insert(tk.END, f"{self.variables[i]} = {solucion[i]}\n")
        
        self.verificar_solucion(matriz, vector, solucion)
        return self.resultado_text.get("1.0", tk.END)

    def mostrar_matriz(self, matriz, titulo=""):
        if titulo:
            self.resultado_text.insert(tk.END, titulo + "\n")
        
        n = matriz.shape[0]
        for i in range(n):
            fila = "| "
            for j in range(matriz.shape[1]):
                val = matriz[i, j]
                if isinstance(val, (int, float)):
                    frac = Fraction(val).limit_denominator()
                    if frac.denominator == 1:
                        fila += f"{frac.numerator:5} "
                    else:
                        fila += f"{frac.numerator}/{frac.denominator:4} "
                else:
                    fila += f"{val:5} "
            fila += "|"
            self.resultado_text.insert(tk.END, fila + "\n")
        self.resultado_text.insert(tk.END, "\n")

    def verificar_solucion(self, matriz, vector, solucion):
        self.resultado_text.insert(tk.END, "\nVerificación:\n")
        n = len(vector)
        
        for i in range(n):
            suma = 0
            ecuacion = ""
            for j in range(n):
                suma += matriz[i, j] * solucion[j]
                ecuacion += f"{matriz[i, j]}*{solucion[j]} + "
            self.resultado_text.insert(tk.END, ecuacion[:-3] + f"= {suma} (debería ser {vector[i]})\n")
        
        if np.allclose(np.dot(matriz, solucion), vector):
            self.resultado_text.insert(tk.END, "\n¡Solución correcta!\n")

    def mostrar_historial(self):
        if not self.historial:
            messagebox.showinfo("Historial", "No hay sistemas resueltos aún.")
            return
            
        hist_ventana = tk.Toplevel(self.root)
        hist_ventana.title("Historial Completo")
        hist_ventana.geometry("800x600")
        
        contenedor = ttk.Frame(hist_ventana)
        contenedor.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(contenedor)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        texto_hist = tk.Text(contenedor, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                           font=('Arial', 10), padx=10, pady=10)
        texto_hist.pack(fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=texto_hist.yview)
        
        texto_hist.insert(tk.END, "HISTORIAL DE SISTEMAS RESUELTOS\n\n")
        for idx, item in enumerate(reversed(self.historial), 1):
            texto_hist.insert(tk.END, f"=== Sistema {idx} ===\n")
            texto_hist.insert(tk.END, f"Tamaño: {item['tamano']}x{item['tamano']}\n")
            texto_hist.insert(tk.END, f"Método: {item['metodo']}\n\n")
            texto_hist.insert(tk.END, "Ecuaciones:\n")
            texto_hist.insert(tk.END, item['sistema'] + "\n")
            texto_hist.insert(tk.END, "Solución:\n")
            texto_hist.insert(tk.END, item['solucion'] + "\n")
            texto_hist.insert(tk.END, "-"*50 + "\n\n")
        
        texto_hist.config(state=tk.DISABLED)
        
        ttk.Button(hist_ventana, text="Cerrar", command=hist_ventana.destroy).pack(pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.configure('Dark.TLabelframe', background='#2d2d2d', foreground='white')
    app = SistemaEcuacionesApp(root)
    root.mainloop()
