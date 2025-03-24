import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import time
from datetime import datetime, timedelta
import copy

# Configuração de seed para reprodutibilidade
np.random.seed(42)
random.seed(42)

class Cliente:
    def __init__(self, id, x, y, demanda, inicio_janela, fim_janela, prioridade=1):
        self.id = id
        self.x = x
        self.y = y
        self.demanda = demanda  # Quantidade a ser entregue
        self.inicio_janela = inicio_janela  # Horário mais cedo para entrega (minutos desde o início do dia)
        self.fim_janela = fim_janela  # Horário mais tarde para entrega
        self.prioridade = prioridade  # Prioridade do cliente (1-5, onde 5 é mais prioritário)
    
    def __str__(self):
        return f"Cliente {self.id}: ({self.x}, {self.y}), Demanda: {self.demanda}, Janela: {self.inicio_janela}-{self.fim_janela}, Prioridade: {self.prioridade}"

class Veiculo:
    def __init__(self, id, capacidade, consumo_base, tipo):
        self.id = id
        self.capacidade = capacidade  # Capacidade máxima
        self.consumo_base = consumo_base  # Consumo de combustível por km (base)
        self.tipo = tipo  # Tipo de veículo (para restrições de área)
    
    def calcular_consumo(self, distancia, carga_atual):
        # Consumo aumenta com a carga
        fator_carga = 1 + (carga_atual / self.capacidade) * 0.5
        return self.consumo_base * distancia * fator_carga
    
    def __str__(self):
        return f"Veículo {self.id}: Capacidade: {self.capacidade}, Consumo Base: {self.consumo_base}, Tipo: {self.tipo}"

class Deposito:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Problema:
    def __init__(self, num_clientes=200, num_veiculos=15, tamanho_mapa=100):
        self.tamanho_mapa = tamanho_mapa
        self.deposito = Deposito(tamanho_mapa/2, tamanho_mapa/2)
        self.clientes = self.gerar_clientes(num_clientes)
        self.veiculos = self.gerar_veiculos(num_veiculos)
        self.matriz_distancias = self.calcular_matriz_distancias()
        
    def gerar_clientes(self, num_clientes):
        clientes = []
        for i in range(num_clientes):
            x = random.uniform(0, self.tamanho_mapa)
            y = random.uniform(0, self.tamanho_mapa)
            demanda = random.randint(1, 20)
            
            # Ajuda da IA para definir as janelas de tempo (em minutos desde o início do dia, 8:00 = 480, 18:00 = 1080)
            inicio_base = random.randint(480, 960)  # Entre 8:00 e 16:00
            duracao = random.randint(60, 240)  # Janela de 1 a 4 horas
            inicio_janela = inicio_base
            fim_janela = inicio_base + duracao
            
            # Alguns clientes têm prioridade maior
            prioridade = random.choices([1, 2, 3, 4, 5], weights=[0.25, 0.5, 1.0, 1.5, 2.0])[0]
            
            clientes.append(Cliente(i, x, y, demanda, inicio_janela, fim_janela, prioridade))
        return clientes
    
    def gerar_veiculos(self, num_veiculos):
        veiculos = []
        tipos = ["pequeno", "medio", "grande"]
        for i in range(num_veiculos):
            tipo = random.choice(tipos)
            if tipo == "pequeno":
                capacidade = random.randint(25, 50)
                consumo_base = 0.5  # Litros por km
            elif tipo == "medio":
                capacidade = random.randint(50, 100)
                consumo_base = 1
            else:  # grande
                capacidade = random.randint(100, 150)
                consumo_base = 1.5
            
            veiculos.append(Veiculo(i, capacidade, consumo_base, tipo))
        return veiculos
    
    def calcular_distancia(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def calcular_matriz_distancias(self):
        n = len(self.clientes) + 1  # +1 para o depósito
        matriz_distancias = np.zeros((n, n))
        
        for i, cliente in enumerate(self.clientes):
            dist = self.calcular_distancia(self.deposito.x, self.deposito.y, cliente.x, cliente.y)
            matriz_distancias[0, i+1] = dist
            matriz_distancias[i+1, 0] = dist
        
        for i, cliente1 in enumerate(self.clientes):
            for j, cliente2 in enumerate(self.clientes):
                if i != j:
                    dist = self.calcular_distancia(cliente1.x, cliente1.y, cliente2.x, cliente2.y)
                    matriz_distancias[i+1, j+1] = dist
        
        return matriz_distancias

class Rota:
    def __init__(self, veiculo, clientes=None):
        self.veiculo = veiculo
        self.clientes = clientes if clientes is not None else []
        self.distancia_total = 0
        self.carga_total = 0
        self.horarios_chegada = {} 
        self.consumo_combustivel = 0
        self.violacoes_janela = 0
        self.violacoes_capacidade = 0
    
    def adicionar_cliente(self, cliente):
        self.clientes.append(cliente)
        self.carga_total += cliente.demanda
    
    def __str__(self):
        return f"Rota {self.veiculo.id}: {[c.id for c in self.clientes]}, Dist: {self.distancia_total:.2f}, Carga: {self.carga_total}"

class Solucao:
    def __init__(self, problema, rotas=None):
        self.problema = problema
        self.rotas = rotas if rotas is not None else []
        self.fitness = float('-inf')
        self.distancia_total = 0
        self.custo_total = 0
        self.veiculos_usados = 0
        self.consumo_total = 0
        self.pontualidade = 0
        
    def avaliar(self):
        self.distancia_total = 0
        self.consumo_total = 0
        self.veiculos_usados = len([r for r in self.rotas if len(r.clientes) > 0])
        total_violacoes_janela = 0
        total_violacoes_capacidade = 0
        total_clientes_atendidos = sum(len(rota.clientes) for rota in self.rotas)
        total_prioridade_atendida = 0
        
        for rota in self.rotas:
            if not rota.clientes:
                continue
                
            # Calcular distância da rota
            distancia_rota = 0
            carga_atual = 0
            tempo_atual = 480  # Começa às 8:00 (em minutos)
            
            # Depósito para primeiro cliente
            if rota.clientes:
                primeiro_cliente = rota.clientes[0]
                dist = self.problema.matriz_distancias[0, primeiro_cliente.id + 1]
                distancia_rota += dist
                tempo_viagem = dist * 2  # Assumindo 30 km/h = 0.5 km/min
                tempo_atual += tempo_viagem
                
                # Verifique violações de janela de tempo
                if tempo_atual < primeiro_cliente.inicio_janela:
                    tempo_atual = primeiro_cliente.inicio_janela
                elif tempo_atual > primeiro_cliente.fim_janela:
                    total_violacoes_janela += (tempo_atual - primeiro_cliente.fim_janela)
                
                rota.horarios_chegada[primeiro_cliente.id] = tempo_atual
                tempo_atual += 15  # 15 minutos para fazer a entrega
                carga_atual += primeiro_cliente.demanda
                total_prioridade_atendida += primeiro_cliente.prioridade
                
                # Verifica violações de capacidade
                if carga_atual > rota.veiculo.capacidade:
                    total_violacoes_capacidade += (carga_atual - rota.veiculo.capacidade)
            
            # Entre clientes
            for i in range(len(rota.clientes) - 1):
                cliente_atual = rota.clientes[i]
                proximo_cliente = rota.clientes[i + 1]
                
                dist = self.problema.matriz_distancias[cliente_atual.id + 1, proximo_cliente.id + 1]
                distancia_rota += dist
                tempo_viagem = dist * 2  # Assumindo 30 km/h
                tempo_atual += tempo_viagem
                
                # Verifique violações de janela de tempo
                if tempo_atual < proximo_cliente.inicio_janela:
                    tempo_atual = proximo_cliente.inicio_janela
                elif tempo_atual > proximo_cliente.fim_janela:
                    total_violacoes_janela += (tempo_atual - proximo_cliente.fim_janela)
                
                rota.horarios_chegada[proximo_cliente.id] = tempo_atual
                tempo_atual += 15  # 15 minutos para fazer a entrega
                carga_atual += proximo_cliente.demanda
                total_prioridade_atendida += proximo_cliente.prioridade
                
                # Verifica violações de capacidade
                if carga_atual > rota.veiculo.capacidade:
                    total_violacoes_capacidade += (carga_atual - rota.veiculo.capacidade)
            
            # Último cliente para depósito
            if rota.clientes:
                ultimo_cliente = rota.clientes[-1]
                dist = self.problema.matriz_distancias[ultimo_cliente.id + 1, 0]
                distancia_rota += dist
            
            rota.distancia_total = distancia_rota
            rota.carga_total = sum(cliente.demanda for cliente in rota.clientes)
            rota.consumo_combustivel = rota.veiculo.calcular_consumo(distancia_rota, rota.carga_total)
            rota.violacoes_janela = total_violacoes_janela
            rota.violacoes_capacidade = total_violacoes_capacidade
            
            self.distancia_total += distancia_rota
            self.consumo_total += rota.consumo_combustivel
        
        # Calcular o fitness
        penalidade_janela = total_violacoes_janela * 10
        penalidade_capacidade = total_violacoes_capacidade * 20
        
        self.custo_total = (
            self.distancia_total * 1.0 +  # Peso da distância
            self.veiculos_usados * 500 +  # Custo fixo por veículo
            self.consumo_total * 5.0 +    # Custo do combustível
            penalidade_janela +           # Penalidade por violações de janela de tempo
            penalidade_capacidade         # Penalidade por violações de capacidade
        )
        
        # Verificar se todos os clientes foram atendidos
        if total_clientes_atendidos < len(self.problema.clientes):
            self.custo_total += (len(self.problema.clientes) - total_clientes_atendidos) * 1000
        
        # Considerar prioridade dos clientes atendidos
        max_prioridade_possivel = sum(cliente.prioridade for cliente in self.problema.clientes)
        if max_prioridade_possivel > 0:
            self.pontualidade = 1.0 - (penalidade_janela / (total_clientes_atendidos * 60))
            self.pontualidade = max(0, min(1, self.pontualidade))
            
            # Calculando fitness final (quanto menor, melhor)
            self.fitness = -self.custo_total  # Negativo porque queremos maximizar o fitness
        
    def __str__(self):
        return f"Solução: {self.veiculos_usados} veículos, Distância: {self.distancia_total:.2f}, Consumo: {self.consumo_total:.2f}L, Custo: {self.custo_total:.2f}, Fitness: {self.fitness:.2f}"

class AlgoritmoGenetico:
    def __init__(self, problema, tamanho_populacao=100, taxa_mutacao=0.2, taxa_crossover=0.7, elitismo=0.1, num_geracoes=100):
        self.problema = problema
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.elitismo = elitismo
        self.num_geracoes = num_geracoes
        self.populacao = []
        self.melhor_solucao = None
        self.historico_fitness = []
        
    def inicializar_populacao(self):
        for _ in range(self.tamanho_populacao):
            solucao = self.gerar_solucao_inicial()
            solucao.avaliar()
            self.populacao.append(solucao)
            
            if self.melhor_solucao is None or solucao.fitness > self.melhor_solucao.fitness:
                self.melhor_solucao = copy.deepcopy(solucao)
    
    def gerar_solucao_inicial(self):
        # Criar uma rota vazia para cada veículo
        rotas = [Rota(veiculo) for veiculo in self.problema.veiculos]
        
        # Distribuir clientes aleatoriamente entre as rotas
        clientes_disponiveis = self.problema.clientes.copy()
        random.shuffle(clientes_disponiveis)
        
        for cliente in clientes_disponiveis:
            # Escolher uma rota aleatória
            rota_idx = random.randint(0, len(rotas) - 1)
            rotas[rota_idx].adicionar_cliente(cliente)
        
        return Solucao(self.problema, rotas)
    
    def selecao_torneio(self, tamanho_torneio=3):
        """Operador de seleção por torneio"""
        candidatos = random.sample(self.populacao, tamanho_torneio)
        return max(candidatos, key=lambda x: x.fitness)
    
    def selecao_roleta(self):
        """Operador de seleção por roleta (proporcional ao fitness)"""
        # Convertemos os fitness para valores positivos
        fitness_min = min(sol.fitness for sol in self.populacao)
        fitness_ajustados = [sol.fitness - fitness_min + 1 for sol in self.populacao]  # +1 para evitar zeros
        soma_fitness = sum(fitness_ajustados)
        
        # Caso todos os fitness sejam iguais
        if soma_fitness == 0:
            return random.choice(self.populacao)
        
        # Selecionar com base na roleta
        ponto = random.uniform(0, soma_fitness)
        acumulado = 0
        for idx, fitness in enumerate(fitness_ajustados):
            acumulado += fitness
            if acumulado >= ponto:
                return self.populacao[idx]
        
        # Caso algo dê errado, retorna o último
        return self.populacao[-1]
    
    def crossover(self, pai1, pai2):
        if random.random() > self.taxa_crossover:
            return copy.deepcopy(pai1), copy.deepcopy(pai2)
        
        # Criar filhos vazios
        filho1 = Solucao(self.problema, [Rota(v) for v in self.problema.veiculos])
        filho2 = Solucao(self.problema, [Rota(v) for v in self.problema.veiculos])
        
        # Criar listas de todos os clientes de ambos os pais
        clientes_pai1 = []
        clientes_pai2 = []
        
        for rota in pai1.rotas:
            clientes_pai1.extend(rota.clientes)
        
        for rota in pai2.rotas:
            clientes_pai2.extend(rota.clientes)
        
        # Verificar se todos os clientes estão presentes
        todos_clientes = set(c.id for c in self.problema.clientes)
        clientes_encontrados = set(c.id for c in clientes_pai1)
        
        # Se alguns clientes estiverem faltando, adicioná-los
        if len(clientes_encontrados) < len(todos_clientes):
            for cliente in self.problema.clientes:
                if cliente.id not in clientes_encontrados:
                    # Adicionar cliente a uma rota aleatória do pai1
                    rota_aleatoria = random.choice(pai1.rotas)
                    rota_aleatoria.adicionar_cliente(cliente)
                    clientes_pai1.append(cliente)
        
        # Ponto de corte aleatório
        ponto_corte = random.randint(1, len(self.problema.clientes) - 1)
        
        # Dividir clientes para os filhos
        clientes_filho1 = clientes_pai1[:ponto_corte] + [c for c in clientes_pai2 if c.id not in [cliente.id for cliente in clientes_pai1[:ponto_corte]]]
        clientes_filho2 = clientes_pai2[:ponto_corte] + [c for c in clientes_pai1 if c.id not in [cliente.id for cliente in clientes_pai2[:ponto_corte]]]
        
        # Distribuir clientes pelas rotas dos filhos (estratégia gulosa)
        for cliente in clientes_filho1:
            # Encontrar a rota com menor carga
            rota_minima = min(filho1.rotas, key=lambda r: r.carga_total)
            rota_minima.adicionar_cliente(cliente)
        
        for cliente in clientes_filho2:
            # Encontrar a rota com menor carga
            rota_minima = min(filho2.rotas, key=lambda r: r.carga_total)
            rota_minima.adicionar_cliente(cliente)
        
        # Avaliar os novos filhos
        filho1.avaliar()
        filho2.avaliar()
        
        return filho1, filho2
    
    def mutacao_troca(self, solucao):
        """Operador de mutação que troca dois clientes de rota"""
        if random.random() > self.taxa_mutacao:
            return
        
        # Encontrar rotas não vazias
        rotas_nao_vazias = [r for r in solucao.rotas if len(r.clientes) > 0]
        if len(rotas_nao_vazias) < 2:
            return
        
        # Selecionar duas rotas diferentes
        rota1, rota2 = random.sample(rotas_nao_vazias, 2)
        
        if not rota1.clientes or not rota2.clientes:
            return
        
        # Selecionar um cliente de cada rota
        cliente1 = random.choice(rota1.clientes)
        cliente2 = random.choice(rota2.clientes)
        
        # Trocar os clientes
        rota1.clientes.remove(cliente1)
        rota2.clientes.remove(cliente2)
        rota1.clientes.append(cliente2)
        rota2.clientes.append(cliente1)
        
        # Atualizar cargas das rotas
        rota1.carga_total = sum(c.demanda for c in rota1.clientes)
        rota2.carga_total = sum(c.demanda for c in rota2.clientes)
    
    
    
    def executar(self):
        # Inicializar população
        print("Inicializando população...")
        self.inicializar_populacao()
        
        # Evolução por gerações
        print(f"Iniciando evolução por {self.num_geracoes} gerações")
        for geracao in range(self.num_geracoes):
            nova_populacao = []
            
            # Elitismo - manter as melhores soluções
            num_elites = int(self.tamanho_populacao * self.elitismo)
            elites = sorted(self.populacao, key=lambda x: x.fitness, reverse=True)[:num_elites]
            nova_populacao.extend(copy.deepcopy(elite) for elite in elites)
            
            # Gerar o resto da população por seleção, crossover e mutação
            while len(nova_populacao) < self.tamanho_populacao:
                # Seleção
                pai1 = self.selecao_torneio()
                pai2 = self.selecao_torneio()
                
                # Crossover
                filho1, filho2 = self.crossover(pai1, pai2)
                
                # Mutação
                self.mutacao_troca(filho1)
                self.mutacao_troca(filho2)
                
                filho1.avaliar()
                filho2.avaliar()
                
                nova_populacao.append(filho1)
                if len(nova_populacao) < self.tamanho_populacao:
                    nova_populacao.append(filho2)
            
            self.populacao = nova_populacao
            
            melhor_atual = max(self.populacao, key=lambda x: x.fitness)
            if melhor_atual.fitness > self.melhor_solucao.fitness:
                self.melhor_solucao = copy.deepcopy(melhor_atual)
            
            self.historico_fitness.append(self.melhor_solucao.fitness)
            
            print(f"Geração {geracao+1}: Melhor fitness = {self.melhor_solucao.fitness:.2f}, "
                    f"Veículos = {self.melhor_solucao.veiculos_usados}, "
                    f"Distância = {self.melhor_solucao.distancia_total:.2f}, "
                    f"Custo = {self.melhor_solucao.custo_total:.2f}")
        
        return self.melhor_solucao
    
    def plotar_historico(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.historico_fitness) + 1), [-f for f in self.historico_fitness])  # Inverter para visualizar minimização
        plt.title('Gráfico de custo total')
        plt.xlabel('Geração')
        plt.ylabel('Custo Total')
        plt.grid(True)
        plt.show()
    

def main():
    print("Configurando problema...")
    problema = Problema(num_clientes=100, num_veiculos=20) 
    
    print("Inicializando algoritmo genético...")
    ag = AlgoritmoGenetico(
        problema, 
        tamanho_populacao=100,
        taxa_mutacao=0.3,
        taxa_crossover=0.7,
        elitismo=0.3,
        num_geracoes=100
    )
    
    print("Executando algoritmo genético...")
    inicio = time.time()
    melhor_solucao = ag.executar()
    fim = time.time()
    
    print("\nResultados finais:")
    print(f"Tempo de execução: {fim - inicio:.2f} segundos")
    print(melhor_solucao)
    print(f"Número de veículos utilizados: {melhor_solucao.veiculos_usados}")
    ag.plotar_historico()


if __name__ == "__main__":
    main()