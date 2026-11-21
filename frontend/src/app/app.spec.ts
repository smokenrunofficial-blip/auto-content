import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
// import { provideHttpClientTesting } from '@angular/common/http/testing';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    TestBed.inject(HttpTestingController)
      .expectOne('/api/youtube/auth/status')
      .flush({ connected: false, cookieCount: 0 });
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render the video search form', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    TestBed.inject(HttpTestingController)
      .expectOne('/api/youtube/auth/status')
      .flush({ connected: true, cookieCount: 2 });
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('Trova video per la tua nicchia');
    expect(compiled.querySelector('input[name="keyword"]')).not.toBeNull();
    expect(compiled.textContent).toContain('Login YouTube (Cache)');
    expect(compiled.textContent).toContain('Connesso');
    expect(compiled.querySelector('button[type="submit"]')?.textContent).toContain(
      'Cerca, scarica e crea Short',
    );
  });

  it('should open the cookie import panel', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const httpTesting = TestBed.inject(HttpTestingController);
    httpTesting.expectOne('/api/youtube/auth/status').flush({ connected: false, cookieCount: 0 });
    fixture.detectChanges();

    const button = [...(fixture.nativeElement as HTMLElement).querySelectorAll('button')].find(
      (candidate) => candidate.textContent?.includes('Login YouTube (Cache)'),
    );
    button?.click();
    fixture.detectChanges();

    expect((fixture.nativeElement as HTMLElement).textContent).toContain(
      'Importa sessione dal Browser',
    );
    httpTesting.expectOne('/api/youtube/auth/status').flush({ connected: false, cookieCount: 0 });
  });
});
