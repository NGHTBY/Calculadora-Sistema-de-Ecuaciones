import numpy as np
from tkinter import *
from tkinter import messagebox, filedialog, simpledialog
import json
import os
from fractions import Fraction

class CalculadoraSistemasEcuaciones:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Sistemas de Ecuaciones Lineales")
        self.root.geometry("900x750")  # Aumenté ligeramente la altura
        
        # Variables
        self.tamanio_sistema = IntVar(value=2)
        self.variables = ['x', 'y', 'z', 'w', 'v', 'u']
        self.entradas = []
        self.resultados = []
        self.metodo = StringVar(value="Gauss")
        
        # Crear interfaz
        self.crear_interfaz()
        
    def crear_interfaz(self):
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True)

        canvas = Canvas(main_frame)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
        scrollbar.pack(side=RIGHT, fill=Y)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        inner_frame = Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        contenedor_horizontal = Frame(inner_frame)
        contenedor_horizontal.pack(fill=BOTH, expand=True)

        contenido_izquierdo = Frame(contenedor_horizontal)
        contenido_izquierdo.pack(side=LEFT, fill=BOTH, expand=True)

        frame_botones = Frame(contenedor_horizontal, padx=10, pady=10)
        frame_botones.pack(side=RIGHT, fill=Y)

        Button(frame_botones, text="Resolver Sistema", bg="green", fg="white",
               command=self.resolver_sistema).pack(fill=X, pady=2)
        Button(frame_botones, text="Limpiar Sistema", bg="red", fg="white",
               command=self.limpiar_sistema).pack(fill=X, pady=2)
        Button(frame_botones, text="Guardar Sistema", bg="blue", fg="white",
               command=self.guardar_sistema).pack(fill=X, pady=2)
        Button(frame_botones, text="Buscar Sistema", bg="yellow", fg="red",
               command=self.buscar_sistema).pack(fill=X, pady=2)
        Button(frame_botones, text="Ayuda", command=self.mostrar_ayuda).pack(fill=X, pady=2)

        frame_tamanio = LabelFrame(contenido_izquierdo, text="Tamaño del sistema", padx=5, pady=5)
        frame_tamanio.pack(padx=10, pady=5, fill=X)

        for i in range(2, 7):
            Radiobutton(frame_tamanio, text=f"{i}x{i}", variable=self.tamanio_sistema,
                        value=i, command=self.actualizar_interfaz).pack(side=LEFT, padx=5)

        self.frame_metodo = LabelFrame(contenido_izquierdo, text="Método de solución", padx=5, pady=5)
        self.frame_metodo.pack(padx=10, pady=5, fill=X)

        self.frame_sistema = LabelFrame(contenido_izquierdo, text="Sistema de ecuaciones", padx=5, pady=5)
        self.frame_sistema.pack(padx=10, pady=5, fill=BOTH, expand=True)

        self.frame_resultados = LabelFrame(contenido_izquierdo, text="Resultados y pasos", padx=5, pady=5)
        self.frame_resultados.pack(padx=10, pady=5, fill=BOTH, expand=True)

        scrollbar_resultados = Scrollbar(self.frame_resultados)
        scrollbar_resultados.pack(side=RIGHT, fill=Y)

        self.texto_resultados = Text(self.frame_resultados, yscrollcommand=scrollbar_resultados.set, wrap=WORD)
        self.texto_resultados.pack(fill=BOTH, expand=True)
        scrollbar_resultados.config(command=self.texto_resultados.yview)

        self.actualizar_interfaz()
    
    def actualizar_interfaz(self):
        # Limpiar frames
        for widget in self.frame_metodo.winfo_children():
            widget.destroy()
        
        for widget in self.frame_sistema.winfo_children():
            widget.destroy()
        
        self.entradas = []
        self.texto_resultados.delete(1.0, END)
        
        # Actualizar métodos según tamaño
        n = self.tamanio_sistema.get()
        
        if n == 2:
            Label(self.frame_metodo, text="Método:").pack(side=LEFT)
            Radiobutton(self.frame_metodo, text="Eliminación", variable=self.metodo, 
                        value="Eliminacion").pack(side=LEFT, padx=5)
            Radiobutton(self.frame_metodo, text="Igualación", variable=self.metodo, 
                        value="Igualacion").pack(side=LEFT, padx=5)
            Radiobutton(self.frame_metodo, text="Sustitución", variable=self.metodo, 
                        value="Sustitucion").pack(side=LEFT, padx=5)
            Radiobutton(self.frame_metodo, text="Gauss", variable=self.metodo, 
                        value="Gauss").pack(side=LEFT, padx=5)
            Radiobutton(self.frame_metodo, text="Gauss-Jordan", variable=self.metodo, 
                        value="Gauss-Jordan").pack(side=LEFT, padx=5)
        else:
            Label(self.frame_metodo, text="Método:").pack(side=LEFT)
            Radiobutton(self.frame_metodo, text="Gauss", variable=self.metodo, 
                        value="Gauss").pack(side=LEFT, padx=5)
            Radiobutton(self.frame_metodo, text="Gauss-Jordan", variable=self.metodo, 
                        value="Gauss-Jordan").pack(side=LEFT, padx=5)
        
        # Crear matriz de entrada
        for i in range(n):
            fila_entradas = []
            for j in range(n + 1):
                entrada = Entry(self.frame_sistema, width=5)
                entrada.grid(row=i, column=j*2, padx=2, pady=2)
                
                # Agregar etiquetas de variables
                if j < n:
                    Label(self.frame_sistema, text=self.variables[j]).grid(row=i, column=j*2+1, padx=2)
                else:
                    Label(self.frame_sistema, text="=").grid(row=i, column=j*2-1, padx=2)
                
                fila_entradas.append(entrada)
            self.entradas.append(fila_entradas)
    
    def limpiar_sistema(self):
        for fila in self.entradas:
            for entrada in fila:
                entrada.delete(0, END)
        self.texto_resultados.delete(1.0, END)
    
    def obtener_matrices(self):
        n = self.tamanio_sistema.get()
        A = []
        B = []
        
        for i in range(n):
            fila_A = []
            for j in range(n):
                try:
                    valor = float(Fraction(self.entradas[i][j].get()))
                except ValueError:
                    messagebox.showerror("Error", f"Valor inválido en la posición {i+1},{j+1}")
                    return None, None
                fila_A.append(valor)
            
            try:
                valor_b = float(Fraction(self.entradas[i][n].get()))
            except ValueError:
                messagebox.showerror("Error", f"Valor inválido en el término independiente de la ecuación {i+1}")
                return None, None
            
            A.append(fila_A)
            B.append(valor_b)
        
        return np.array(A, dtype=float), np.array(B, dtype=float)
    
    def resolver_sistema(self):
        A, B = self.obtener_matrices()
        if A is None or B is None:
            return
        
        n = self.tamanio_sistema.get()
        metodo = self.metodo.get()
        
        self.texto_resultados.delete(1.0, END)
        self.texto_resultados.insert(END, f"Resolviendo sistema {n}x{n} por método de {metodo}:\n\n")
        
        try:
            if metodo in ["Gauss", "Gauss-Jordan"]:
                self.resolver_por_gauss(A, B, metodo)
            elif metodo == "Eliminacion":
                self.resolver_2x2_eliminacion(A, B)
            elif metodo == "Igualacion":
                self.resolver_2x2_igualacion(A, B)
            elif metodo == "Sustitucion":
                self.resolver_2x2_sustitucion(A, B)
        except np.linalg.LinAlgError as e:
            self.texto_resultados.insert(END, f"\nError: {str(e)}")
        except Exception as e:
            self.texto_resultados.insert(END, f"\nError inesperado: {str(e)}")
            messagebox.showerror("Error", f"Ocurrió un error inesperado: {str(e)}")
    
    def resolver_por_gauss(self, A, B, metodo):
        n = len(A)
        pasos = []
        
        # Matriz aumentada
        M = np.column_stack((A, B))
        pasos.append(("Matriz aumentada inicial:", M.copy()))
        
        # Eliminación hacia adelante
        rank = 0
        for i in range(n):
            # Pivoteo parcial
            max_row = np.argmax(abs(M[i:, i])) + i
            if max_row != i:
                M[[i, max_row]] = M[[max_row, i]]
                pasos.append((f"Intercambio fila {i+1} con fila {max_row+1}:", M.copy()))
            
            # Verificar si el pivote es cero
            if abs(M[i, i]) < 1e-10:
                # Verificar si toda la fila es cero
                if np.all(abs(M[i, :-1]) < 1e-10):
                    if abs(M[i, -1]) < 1e-10:
                        # Fila de ceros - sistema puede tener infinitas soluciones
                        continue
                    else:
                        # Sistema incompatible
                        self.texto_resultados.insert(END, "\nEl sistema no tiene solución (fila de ceros con término independiente no nulo).")
                        return
                else:
                    # Sistema puede tener infinitas soluciones
                    continue
            
            rank += 1
            
            # Hacer ceros debajo del pivote
            for j in range(i + 1, n):
                factor = M[j, i] / M[i, i]
                M[j, i:] -= factor * M[i, i:]
                pasos.append((f"F{j+1} = F{j+1} - ({self.formatear_numero(factor)})*F{i+1}:", M.copy()))
        
        # Verificar rango para infinitas soluciones
        if rank < n:
            # Verificar consistencia
            for i in range(rank, n):
                if abs(M[i, -1]) > 1e-10 and not np.all(abs(M[i, :-1]) < 1e-10):
                    self.texto_resultados.insert(END, "\nEl sistema no tiene solución.")
                    return
            
            self.texto_resultados.insert(END, "\nEl sistema tiene infinitas soluciones (rango < número de variables).")
            self.manejar_infinitas_soluciones(M)
            return
        
        # Si es Gauss-Jordan, hacer ceros también arriba del pivote
        if metodo == "Gauss-Jordan":
            for i in range(n-1, -1, -1):
                for j in range(i-1, -1, -1):
                    factor = M[j, i] / M[i, i]
                    M[j, i:] -= factor * M[i, i:]
                    pasos.append((f"F{j+1} = F{j+1} - ({self.formatear_numero(factor)})*F{i+1}:", M.copy()))
        
        # Mostrar pasos
        for descripcion, matriz in pasos:
            self.texto_resultados.insert(END, f"\n{descripcion}\n")
            self.texto_resultados.insert(END, self.formatear_matriz(matriz))
        
        # Sustitución hacia atrás solo si no hay infinitas soluciones
        soluciones = np.zeros(n)
        for i in range(n-1, -1, -1):
            if abs(M[i, i]) < 1e-10:
                if abs(M[i, -1]) > 1e-10:
                    self.texto_resultados.insert(END, "\nEl sistema no tiene solución.")
                    return
                else:
                    self.texto_resultados.insert(END, "\nEl sistema tiene infinitas soluciones.")
                    self.manejar_infinitas_soluciones(M)
                    return
            
            soluciones[i] = (M[i, -1] - np.dot(M[i, i+1:n], soluciones[i+1:n])) / M[i, i]
        
        # Mostrar resultados
        self.texto_resultados.insert(END, "\nSolución del sistema:\n")
        for i in range(n):
            sol_fraccion = Fraction(soluciones[i]).limit_denominator()
            if sol_fraccion.denominator == 1:
                self.texto_resultados.insert(END, f"{self.variables[i]} = {sol_fraccion.numerator}\n")
            else:
                self.texto_resultados.insert(END, f"{self.variables[i]} = {sol_fraccion.numerator}/{sol_fraccion.denominator}\n")
        
        # Comprobar solución
        self.comprobar_solucion(A, B, soluciones)
    
    def resolver_2x2_sustitucion(self, A, B):
        self.texto_resultados.insert(END, "\nMétodo de Sustitución:\n")
        
        # Paso 1: Mostrar ecuaciones originales
        self.texto_resultados.insert(END, "\nEcuaciones originales:\n")
        self.texto_resultados.insert(END, f"{A[0,0]}x + {A[0,1]}y = {B[0]}\n")
        self.texto_resultados.insert(END, f"{A[1,0]}x + {A[1,1]}y = {B[1]}\n")
    
        a, b = A[0, 0], A[0, 1]
        c = B[0]
        d, e = A[1, 0], A[1, 1]
        f = B[1]
        
        # Paso 1: Despejar x en la primera ecuación
        self.texto_resultados.insert(END, "\nPaso 1: Despejamos x en la primera ecuación:\n")
        self.texto_resultados.insert(END, f"{a}x + {b}y = {c}\n")
        self.texto_resultados.insert(END, f"x = ({c} - {b}y) / {a}\n")
        
        # Paso 2: Sustituir en la segunda ecuación
        self.texto_resultados.insert(END, "\nPaso 2: Sustituimos x en la segunda ecuación:\n")
        self.texto_resultados.insert(END, f"{d}*(({c} - {b}y)/{a}) + {e}y = {f}\n")
        
        # Simplificar
        term1 = (d*c)/a
        term2 = (d*b)/a
        self.texto_resultados.insert(END, f"({term1} - {term2}y) + {e}y = {f}\n")
        
        # Resolver para y
        coef_y = e - (d*b)/a
        const = f - (d*c)/a
        self.texto_resultados.insert(END, f"\nAgrupando términos para y:\n")
        self.texto_resultados.insert(END, f"{coef_y}y = {const}\n")
        
        try:
            y = const / coef_y
            y_frac = Fraction(y).limit_denominator()
            self.texto_resultados.insert(END, f"y = {const} / {coef_y} = {y_frac}\n")
            
            # Paso 3: Sustituir y para encontrar x
            self.texto_resultados.insert(END, "\nPaso 3: Sustituimos y en la ecuación despejada de x:\n")
            x = (c - b*y) / a
            x_frac = Fraction(x).limit_denominator()
            self.texto_resultados.insert(END, f"x = ({c} - {b}*{y_frac}) / {a} = {x_frac}\n")
            
            # Mostrar solución
            self.texto_resultados.insert(END, "\nSolución del sistema:\n")
            self.texto_resultados.insert(END, f"x = {x_frac}\n")
            self.texto_resultados.insert(END, f"y = {y_frac}\n")
            
            # Comprobar solución
            self.comprobar_solucion(A, B, [x, y])
            
        except ZeroDivisionError:
            self.texto_resultados.insert(END, "\nError: División por cero. El sistema no tiene solución única.\n")
            if abs(const) < 1e-10:
                self.texto_resultados.insert(END, "El sistema tiene infinitas soluciones.\n")
                self.manejar_infinitas_soluciones(np.column_stack((A, B)))
            else:
                self.texto_resultados.insert(END, "El sistema no tiene solución.\n")

    def resolver_2x2_eliminacion(self, A, B):
        self.texto_resultados.insert(END, "\nMétodo de Eliminación:\n")
        
        # Paso 1: Mostrar ecuaciones originales
        self.texto_resultados.insert(END, "\nEcuaciones originales:\n")
        self.texto_resultados.insert(END, f"(1) {A[0,0]}x + {A[0,1]}y = {B[0]}\n")
        self.texto_resultados.insert(END, f"(2) {A[1,0]}x + {A[1,1]}y = {B[1]}\n")
        
        a, b = A[0, 0], A[0, 1]
        c = B[0]
        d, e = A[1, 0], A[1, 1]
        f = B[1]
        
        try:
            # Paso 2: Eliminar x de la segunda ecuación
            self.texto_resultados.insert(END, "\nPaso 1: Eliminar x de la segunda ecuación\n")
            
            # Calcular el factor de multiplicación
            factor = d / a
            self.texto_resultados.insert(END, f"Multiplicamos la ecuación (1) por {d}/{a} para igualar coeficientes de x:\n")
            
            # Mostrar ecuación modificada
            self.texto_resultados.insert(END, f"({d}/{a}) * ({a}x + {b}y = {c})\n")
            self.texto_resultados.insert(END, f"-> {d}x + {(d*b)/a}y = {(d*c)/a}\n")
            
            # Restar de la segunda ecuación
            self.texto_resultados.insert(END, f"\nRestamos esta nueva ecuación de la ecuación (2):\n")
            self.texto_resultados.insert(END, f"({d}x + {e}y = {f}) - ({d}x + {(d*b)/a}y = {(d*c)/a})\n")
            
            # Calcular nuevos coeficientes para y
            nuevo_coef_y = e - (d*b)/a
            nuevo_const = f - (d*c)/a
            self.texto_resultados.insert(END, f"-> (0x + {nuevo_coef_y}y = {nuevo_const})\n")
            
            # Resolver para y
            self.texto_resultados.insert(END, f"\nPaso 2: Resolver para y\n")
            y = nuevo_const / nuevo_coef_y
            y_frac = Fraction(y).limit_denominator()
            self.texto_resultados.insert(END, f"y = {nuevo_const} / {nuevo_coef_y} = {y_frac}\n")
            
            # Paso 3: Sustituir y para encontrar x
            self.texto_resultados.insert(END, "\nPaso 3: Sustituir y en la primera ecuación para encontrar x\n")
            self.texto_resultados.insert(END, f"Sustituimos y = {y_frac} en la ecuación (1):\n")
            self.texto_resultados.insert(END, f"{a}x + {b}*{y_frac} = {c}\n")
            
            x = (c - b*y) / a
            x_frac = Fraction(x).limit_denominator()
            self.texto_resultados.insert(END, f"x = ({c} - {b*y_frac}) / {a} = {x_frac}\n")
            
            # Mostrar solución
            self.texto_resultados.insert(END, "\nSolución del sistema:\n")
            self.texto_resultados.insert(END, f"x = {x_frac}\n")
            self.texto_resultados.insert(END, f"y = {y_frac}\n")
            
            # Comprobar solución
            self.comprobar_solucion(A, B, [x, y])
            
        except ZeroDivisionError:
            self.texto_resultados.insert(END, "\nError: División por cero. El sistema no tiene solución única.\n")
            if abs(nuevo_const) < 1e-10:
                self.texto_resultados.insert(END, "El sistema tiene infinitas soluciones.\n")
                self.manejar_infinitas_soluciones(np.column_stack((A, B)))
            else:
                self.texto_resultados.insert(END, "El sistema no tiene solución.\n")             

    def resolver_2x2_igualacion(self, A, B):
        self.texto_resultados.insert(END, "\nMétodo de Igualación:\n")
    
        # Paso 1: Mostrar ecuaciones originales
        self.texto_resultados.insert(END, "\nEcuaciones originales:\n")
        self.texto_resultados.insert(END, f"(1) {A[0,0]}x + {A[0,1]}y = {B[0]}\n")
        self.texto_resultados.insert(END, f"(2) {A[1,0]}x + {A[1,1]}y = {B[1]}\n")
    
        a, b = A[0, 0], A[0, 1]
        c = B[0]
        d, e = A[1, 0], A[1, 1]
        f = B[1]
    
        try:
            # Paso 2: Despejar x en ambas ecuaciones
            self.texto_resultados.insert(END, "\nPaso 1: Despejar x en ambas ecuaciones:\n")
        
            # Despejar x en la primera ecuación
            self.texto_resultados.insert(END, f"\nDe la ecuación (1):\n")
            self.texto_resultados.insert(END, f"{a}x = {c} - {b}y\n")
            self.texto_resultados.insert(END, f"x = ({c} - {b}y)/{a}\n")
            expr_x1 = f"({c} - {b}y)/{a}"
        
            # Despejar x en la segunda ecuación
            self.texto_resultados.insert(END, f"\nDe la ecuación (2):\n")
            self.texto_resultados.insert(END, f"{d}x = {f} - {e}y\n")
            self.texto_resultados.insert(END, f"x = ({f} - {e}y)/{d}\n")
            expr_x2 = f"({f} - {e}y)/{d}"
        
            # Paso 3: Igualar las dos expresiones para x
            self.texto_resultados.insert(END, "\nPaso 2: Igualar las dos expresiones para x:\n")
            self.texto_resultados.insert(END, f"{expr_x1} = {expr_x2}\n")
        
            # Eliminar denominadores
            self.texto_resultados.insert(END, "\nMultiplicar ambos lados por los denominadores:\n")
            self.texto_resultados.insert(END, f"{d}({c} - {b}y) = {a}({f} - {e}y)\n")
        
            # Expandir
            term1 = d*c
            term2 = d*b
            term3 = a*f
            term4 = a*e
            self.texto_resultados.insert(END, f"{term1} - {term2}y = {term3} - {term4}y\n")
        
            # Paso 4: Resolver para y
            self.texto_resultados.insert(END, "\nPaso 3: Resolver para y:\n")
        
            # Mover términos con y a un lado y constantes al otro
            self.texto_resultados.insert(END, f"{term4}y - {term2}y = {term3} - {term1}\n")
        
            # Simplificar
            coef_y = term4 - term2
            const = term3 - term1
            self.texto_resultados.insert(END, f"{coef_y}y = {const}\n")
        
            # Calcular y
            y = const / coef_y
            y_frac = Fraction(y).limit_denominator()
            self.texto_resultados.insert(END, f"y = {const} / {coef_y} = {y_frac}\n")
        
            # Paso 5: Sustituir y en una de las expresiones para x
            self.texto_resultados.insert(END, "\nPaso 4: Sustituir y en una de las expresiones para x:\n")
        
            # Usar la primera expresión para x
            self.texto_resultados.insert(END, f"Usando x = ({c} - {b}y)/{a}\n")
            self.texto_resultados.insert(END, f"x = ({c} - {b}*{y_frac})/{a}\n")
        
            # Calcular x
            x = (c - b*y) / a
            x_frac = Fraction(x).limit_denominator()
            self.texto_resultados.insert(END, f"x = {x_frac}\n")
        
            # Mostrar solución
            self.texto_resultados.insert(END, "\nSolución del sistema:\n")
            self.texto_resultados.insert(END, f"x = {x_frac}\n")
            self.texto_resultados.insert(END, f"y = {y_frac}\n")
        
            # Comprobar solución
            self.comprobar_solucion(A, B, [x, y])
        
        except ZeroDivisionError:
            self.texto_resultados.insert(END, "\nError: División por cero. El sistema no tiene solución única.\n")
            if abs(const) < 1e-10:
                self.texto_resultados.insert(END, "El sistema tiene infinitas soluciones.\n")
                self.manejar_infinitas_soluciones(np.column_stack((A, B)))
            else:
                self.texto_resultados.insert(END, "El sistema no tiene solución.\n")


    def manejar_infinitas_soluciones(self, M):
        try:
            n = self.tamanio_sistema.get()
            m = M.shape[1] - 1  # Número de variables
            
            # Mostrar que el sistema tiene infinitas soluciones
            self.texto_resultados.insert(END, "\n\nEl sistema tiene infinitas soluciones porque:")
            self.texto_resultados.insert(END, "\n- Al menos una fila tiene todos los coeficientes cero")
            self.texto_resultados.insert(END, "\n- El rango de la matriz de coeficientes es menor que el número de variables")
            
            # Encontrar variables pivote (dependientes) y variables libres
            variables_pivote = []
            variables_libres = list(range(m))
            
            for i in range(n):
                if not np.all(abs(M[i, :-1]) < 1e-10):
                    pivot_col = np.where(abs(M[i, :-1]) > 1e-10)[0][0]
                    if pivot_col in variables_libres:
                        variables_libres.remove(pivot_col)
                    if pivot_col not in variables_pivote:
                        variables_pivote.append(pivot_col)
            
            if not variables_libres:
                return
            
            # Mostrar información sobre variables libres
            self.texto_resultados.insert(END, f"\n\nVariables libres: {', '.join([self.variables[j] for j in variables_libres])}")
            self.texto_resultados.insert(END, f"\nVariables dependientes: {', '.join([self.variables[j] for j in variables_pivote])}")
            
            # Pedir valor para una variable libre
            var_libre = variables_libres[0]
            var_nombre = self.variables[var_libre]
            
            respuesta = messagebox.askyesno("Infinitas Soluciones", 
                                          f"El sistema tiene infinitas soluciones. ¿Desea asignar un valor a {var_nombre}?",
                                          parent=self.root)
            
            if respuesta:
                valor = simpledialog.askfloat("Valor de variable", 
                                             f"Ingrese el valor para {var_nombre}:",
                                             parent=self.root)
                
                if valor is not None:
                    # Mostrar el valor asignado
                    self.texto_resultados.insert(END, f"\n\nAsignando {var_nombre} = {valor}")
                    
                    # Resolver el sistema con el valor asignado
                    soluciones = np.zeros(m)
                    soluciones[var_libre] = valor
                    
                    # Sustituir hacia atrás
                    for i in range(n-1, -1, -1):
                        if np.all(abs(M[i, :-1]) < 1e-10):
                            continue
                        
                        pivot_col = np.where(abs(M[i, :-1]) > 1e-10)[0][0]
                        if pivot_col in variables_pivote:
                            term_ind = M[i, -1]
                            suma = 0
                            
                            # Mostrar la ecuación actual
                            self.texto_resultados.insert(END, f"\n\nResolviendo para {self.variables[pivot_col]}:")
                            ecuacion = f"{self.formatear_numero(M[i, pivot_col])}{self.variables[pivot_col]}"
                            for j in range(pivot_col + 1, m):
                                if abs(M[i, j]) > 1e-10:
                                    if M[i, j] > 0:
                                        ecuacion += f" + {self.formatear_numero(M[i, j])}{self.variables[j]}"
                                    else:
                                        ecuacion += f" - {self.formatear_numero(-M[i, j])}{self.variables[j]}"
                                suma += M[i, j] * soluciones[j]
                            
                            ecuacion += f" = {self.formatear_numero(term_ind)}"
                            self.texto_resultados.insert(END, "\n" + ecuacion)
                            
                            # Mostrar sustitución de valores conocidos
                            if pivot_col + 1 < m:
                                self.texto_resultados.insert(END, "\nSustituyendo valores conocidos:")
                                sust_text = ""
                                for j in range(pivot_col + 1, m):
                                    if abs(M[i, j]) > 1e-10:
                                        sust_text += f"\n{self.variables[j]} = {soluciones[j]}"
                                self.texto_resultados.insert(END, sust_text)
                            
                            # Calcular y mostrar el resultado
                            sol = (term_ind - suma) / M[i, pivot_col]
                            soluciones[pivot_col] = sol
                            
                            self.texto_resultados.insert(END, f"\n{self.variables[pivot_col]} = ({self.formatear_numero(term_ind)} - {self.formatear_numero(suma)}) / {self.formatear_numero(M[i, pivot_col])}")
                            self.texto_resultados.insert(END, f"\n{self.variables[pivot_col]} = {self.formatear_numero(sol)}")
                    
                    # Mostrar resultados finales
                    self.texto_resultados.insert(END, "\n\nSolución particular con el valor asignado:")
                    for i in range(m):
                        sol_fraccion = Fraction(soluciones[i]).limit_denominator()
                        if sol_fraccion.denominator == 1:
                            self.texto_resultados.insert(END, f"\n{self.variables[i]} = {sol_fraccion.numerator}")
                        else:
                            self.texto_resultados.insert(END, f"\n{self.variables[i]} = {sol_fraccion.numerator}/{sol_fraccion.denominator}")
                    
                    # Comprobar la solución
                    self.comprobar_solucion(M[:, :-1], M[:, -1], soluciones)
                    
                    # Mostrar solución general
                    self.texto_resultados.insert(END, "\n\nSolución general del sistema (con parámetro libre):")
                    for i in range(m):
                        if i in variables_pivote:
                            # Encontrar la fila que corresponde a esta variable pivote
                            fila_pivote = None
                            for fila in range(n):
                                if not np.all(abs(M[fila, :-1]) < 1e-10) and np.where(abs(M[fila, :-1]) > 1e-10)[0][0] == i:
                                    fila_pivote = fila
                                    break
                            
                            if fila_pivote is not None:
                                ecuacion = f"{self.variables[i]} = {self.formatear_numero(M[fila_pivote, -1])}"
                                for j in variables_libres:
                                    if abs(M[fila_pivote, j]) > 1e-10:
                                        coef = -M[fila_pivote, j] / M[fila_pivote, i]
                                        if coef > 0:
                                            ecuacion += f" + {self.formatear_numero(coef)}{self.variables[j]}"
                                        else:
                                            ecuacion += f" - {self.formatear_numero(-coef)}{self.variables[j]}"
                                self.texto_resultados.insert(END, "\n" + ecuacion)
                        else:
                            self.texto_resultados.insert(END, f"\n{self.variables[i]} = t (parámetro libre)")
        except Exception as e:
            messagebox.showerror("Error", f"Error al manejar infinitas soluciones: {str(e)}")
            self.texto_resultados.insert(END, f"\nError: {str(e)}")
    
    def comprobar_solucion(self, A, B, solucion):
        """Comprueba si la solución satisface el sistema original"""
        self.texto_resultados.insert(END, "\n\nComprobación de la solución:\n")
        
        n = len(A)
        tolerancia = 1e-10
        
        for i in range(n):
            # Calcular el lado izquierdo de la ecuación
            lado_izq = 0.0
            ecuacion = ""
            
            for j in range(len(solucion)):
                coef = A[i, j]
                valor = solucion[j]
                lado_izq += coef * valor
                
                if j > 0 and coef >= 0:
                    ecuacion += " + "
                elif j > 0:
                    ecuacion += " - "
                elif coef < 0:
                    ecuacion += "-"
                
                if abs(coef) != 1:
                    ecuacion += f"{self.formatear_numero(abs(coef))}"
                
                ecuacion += f"{self.variables[j]}"
                
                if abs(valor) != 1:
                    ecuacion += f"({self.formatear_numero(valor)})"
            
            # Mostrar la evaluación
            self.texto_resultados.insert(END, f"Ecuación {i+1}: {ecuacion} = {self.formatear_numero(lado_izq)}\n")
            self.texto_resultados.insert(END, f"Valor esperado: {self.formatear_numero(B[i])}\n")
            
            # Verificar si se cumple la ecuación
            if abs(lado_izq - B[i]) < tolerancia:
                self.texto_resultados.insert(END, "✓ La ecuación se cumple.\n\n")
            else:
                self.texto_resultados.insert(END, f"✗ La ecuación no se cumple. Diferencia: {abs(lado_izq - B[i]):.2e}\n\n")
    
    def formatear_numero(self, num):
        """Formatea un número como fracción si es posible, o con 2 decimales si no"""
        try:
            fraccion = Fraction(num).limit_denominator()
            if fraccion.denominator == 1:
                return str(fraccion.numerator)
            else:
                return f"{fraccion.numerator}/{fraccion.denominator}"
        except:
            return f"{num:.4f}"
    
    def formatear_matriz(self, matriz):
        texto = ""
        for fila in matriz:
            for elemento in fila:
                fraccion = Fraction(elemento).limit_denominator()
                if fraccion.denominator == 1:
                    texto += f"{fraccion.numerator:>8}"
                else:
                    texto += f"{fraccion.numerator}/{fraccion.denominator:>7}"
            texto += "\n"
        return texto
    
    def guardar_sistema(self):
        A, B = self.obtener_matrices()
        if A is None or B is None:
            return
        
        nombre = simpledialog.askstring("Guardar sistema", "Nombre del sistema:", parent=self.root)
        if not nombre:
            return
        
        sistema = {
            'tamanio': self.tamanio_sistema.get(),
            'matriz_A': A.tolist(),
            'vector_B': B.tolist()
        }
        
        # Crear directorio si no existe
        if not os.path.exists("sistemas_guardados"):
            os.makedirs("sistemas_guardados")
        
        # Guardar en archivo JSON
        with open(f"sistemas_guardados/{nombre}.json", 'w') as f:
            json.dump(sistema, f)
        
        messagebox.showinfo("Guardado", f"Sistema '{nombre}' guardado correctamente.")
    
    def buscar_sistema(self):
        # Listar archivos disponibles
        if not os.path.exists("sistemas_guardados"):
            messagebox.showinfo("Buscar", "No hay sistemas guardados.")
            return
        
        archivos = os.listdir("sistemas_guardados")
        if not archivos:
            messagebox.showinfo("Buscar", "No hay sistemas guardados.")
            return
        
        # Mostrar diálogo para seleccionar
        seleccion = simpledialog.askstring("Buscar sistema", 
                                         "Sistemas disponibles:\n" + "\n".join([f[:-5] for f in archivos]) + 
                                         "\n\nIngrese el nombre del sistema a cargar:",
                                         parent=self.root)
        
        if not seleccion:
            return
        
        try:
            with open(f"sistemas_guardados/{seleccion}.json", 'r') as f:
                sistema = json.load(f)
            
            # Configurar tamaño
            self.tamanio_sistema.set(sistema['tamanio'])
            self.actualizar_interfaz()
            
            # Llenar valores
            A = np.array(sistema['matriz_A'])
            B = np.array(sistema['vector_B'])
            n = sistema['tamanio']
            
            for i in range(n):
                for j in range(n):
                    self.entradas[i][j].delete(0, END)
                    self.entradas[i][j].insert(0, str(A[i, j]))
                
                self.entradas[i][n].delete(0, END)
                self.entradas[i][n].insert(0, str(B[i]))
            
            messagebox.showinfo("Cargado", f"Sistema '{seleccion}' cargado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el sistema: {str(e)}")
    
    def mostrar_ayuda(self):
        ayuda_texto = """
        INSTRUCCIONES DE USO:
        
        1. Seleccione el tamaño del sistema (2x2 hasta 6x6).
        2. Ingrese los coeficientes de las ecuaciones en las casillas correspondientes.
        3. Seleccione el método de solución:
           - Para sistemas 2x2: Eliminación, Igualación, Sustitución, Gauss o Gauss-Jordan
           - Para sistemas 3x3 a 6x6: Gauss o Gauss-Jordan
        4. Haga clic en "Resolver Sistema" para ver la solución y los pasos.
        5. Use "Limpiar Sistema" para borrar todos los campos.
        6. Use "Guardar Sistema" para guardar el sistema actual con un nombre.
        7. Use "Buscar Sistema" para cargar un sistema previamente guardado.
        
        NOTAS:
        - Puede ingresar números decimales y negativos.
        - Los resultados se mostrarán como fracciones exactas cuando sea posible.
        - Si el sistema tiene infinitas soluciones, se le pedirá asignar un valor a una variable.
        """
        messagebox.showinfo("Ayuda", ayuda_texto)

if __name__ == "__main__":
    root = Tk()
    app = CalculadoraSistemasEcuaciones(root)
    root.mainloop()
